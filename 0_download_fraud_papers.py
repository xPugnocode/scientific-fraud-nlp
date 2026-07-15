import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib3.util.retry import Retry

IDCONV_URL = 'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/' # https://pmc.ncbi.nlm.nih.gov/tools/id-converter-api/
S3_BASE = 'https://pmc-oa-opendata.s3.amazonaws.com' # https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
ENTREZ_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi' # https://pmc.ncbi.nlm.nih.gov/tools/get-metadata/

session = requests.Session()
retry = Retry(status_forcelist=(429, 503), allowed_methods=("GET"))
adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
session.mount("https://", adapter)

def filter_retractions(big_df):
    small_df = big_df[(
        (big_df['Reason'].str.contains('Falsification') | big_df['Reason'].str.contains('Manipulation'))
        & (big_df['ArticleType'].str.contains('Research Article') | big_df['ArticleType'].str.contains('Clinical Study'))
        & (big_df['Subject'].str.contains('BLS') | big_df['Subject'].str.contains('HSC'))
        & ~(big_df['OriginalPaperPubMedID'].eq(0.0, fill_value=0.0))
        & (big_df['RetractionNature'] == 'Retraction')
    )]
    return small_df


def find_pmcid(filtered_df):
    results = []
    papers = [
        (row['OriginalPaperDOI'], str(int(row['OriginalPaperPubMedID'])))
        for _, row in filtered_df.iterrows()
    ]
    for start in tqdm(range(0, len(papers), 200), desc='Checking PMCID', unit=' batches'): 
        batch = papers[start:start + 200] # maximum limit of 200 papers per request
        # time.sleep(1/3)
        r = session.get(IDCONV_URL, params={'idtype': 'pmid', 'ids': ','.join(pmid for _, pmid in batch), 'format': 'json'})
        r.raise_for_status()
        data = r.json()

        records = {str(record.get('pmid')): record for record in data.get('records', [])}
        for doi, pmid in batch:
            pmcid = records.get(pmid, {}).get('pmcid')
            if pmcid is not None:
                results.append((doi, pmid, pmcid))    
    # print(results)
    # print('Total papers with PMCID:', len(results))
    return results

def check_pmcid_versions(papers):
    def get_versions(paper):
        doi, pmid, pmcid = paper
        r = session.get(S3_BASE, params={'list-type': '2', 'prefix': pmcid, 'delimiter': '/'})
        r.raise_for_status()
        # print(r.content)
        root = ET.fromstring(r.text)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        versions = [p.text for p in root.findall('s3:CommonPrefixes/s3:Prefix', namespaces=ns)]
        if not versions:
            return None
        versions = [version.rstrip('/') for version in versions]

        return doi, pmid, versions

    # each request is independent, so fastest way to speed it up is doing it concurrently

    with ThreadPoolExecutor(max_workers=20) as executor:
        checked_results = tqdm(executor.map(get_versions, papers), total=len(papers), desc='Checking PMCID versions', unit=' papers')
        papers = [result for result in checked_results if result is not None]
        # print('Total papers with downloadable versions:', len(new_results))
    return papers



def get_xml_link(papers):
    def check_xml_availability(paper):
        doi, pmid, versions = paper
        for version in versions:
            pmcid = version.rsplit('.', 1)[0]
            r = session.get(f'{S3_BASE}/metadata/{version}.json')
            try:
                r.raise_for_status()
            except requests.RequestException as e:
                r = session.get(f'{S3_BASE}/{version}/{version}.json')
                try:
                    r.raise_for_status()
                except requests.RequestException as e:
                    continue
            meta = r.json()
            xml_url = meta.get('xml_url')
            if xml_url:
                return str(doi), str(pmid), str(pmcid), str(xml_url)
        print(f'No XML found for PMID: {pmid}')
        return doi, pmid, versions[0].rsplit('.', 1)[0], None

    with ThreadPoolExecutor(max_workers=20) as executor:
        availability = list(tqdm(executor.map(check_xml_availability, papers), total=len(papers), desc='Fetching XML links', unit=' papers'))
    # print(len(availability), availability)
    return pd.DataFrame(availability, columns=['OriginalPaperDOI', 'OriginalPaperPubMedID', 'PMCID', 'XMLLink']).dropna()

def download_papers(papers, papers_dir):
    def download_paper(paper):
        _, _, pmcid, xml_url = paper
        xml_url = xml_url.replace('s3://pmc-oa-opendata', S3_BASE)
        r = session.get(xml_url)
        r.raise_for_status()
        xml_filename = papers_dir / f'{pmcid}.xml'
        xml_filename.write_bytes(r.content)
        
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(download_paper, papers.values), total=len(papers), desc='Downloading papers', unit=' papers'))


def clean_text(element):
    if element is None:
        return None
    return ' '.join(''.join(element.itertext()).split())


def get_more_metadata(fraud_metadata_df, papers_dir, data_dir):
    pubmed_articles = {}
    pmc_articles = {}

    pmids = [str(pmid) for pmid in fraud_metadata_df['OriginalPaperPubMedID']]
    for start in tqdm(range(0, len(pmids), 200), desc='Fetching PubMed metadata', unit=' batches'):
        r = session.get(ENTREZ_URL, params={'db': 'pubmed', 'id': ','.join(pmids[start:start + 200])})
        r.raise_for_status()
        for pubmed_article in ET.fromstring(r.content).findall('PubmedArticle'):
            pubmed_articles[clean_text(pubmed_article.find('MedlineCitation/PMID'))] = pubmed_article

    pmcids = [str(pmcid) for pmcid in fraud_metadata_df['PMCID']]
    for start in tqdm(range(0, len(pmcids), 200), desc='Fetching PMC metadata', unit=' batches'):
        r = session.get(ENTREZ_URL, params={'db': 'pmc', 'id': ','.join(pmcids[start:start + 200]), 'retmode': 'xml'})
        r.raise_for_status()
        for pmc_article in ET.fromstring(r.content).findall('article'):
            pmc_articles[clean_text(pmc_article.find("front/article-meta/article-id[@pub-id-type='pmcid']"))] = pmc_article

    parsed_papers = []
    failed_files = []
    removed_papers = 0
    for row in tqdm(fraud_metadata_df.itertuples(index=False), total=len(fraud_metadata_df), desc='Parsing XML', unit=' papers'):
        doi, pmid, pmcid, xml_link = row
        xml_file = papers_dir / f'{pmcid}.xml'
        try:
            root = ET.parse(xml_file).getroot()
        except (OSError, ET.ParseError):
            failed_files.append(pmcid)
            continue
        article = root.find('article') if root.tag == 'pmc-articleset' else root
        if article is None or article.get('article-type') != 'research-article':
            xml_file.unlink()
            removed_papers += 1
            continue
        pubmed_article = pubmed_articles.get(str(pmid))
        pmc_article = pmc_articles.get(str(pmcid))

        mesh_terms = []
        mesh_headings = pubmed_article.find('MedlineCitation/MeshHeadingList') if pubmed_article is not None else None
        if mesh_headings is not None:
            for mesh_heading in mesh_headings.findall('MeshHeading'):
                descriptor = clean_text(mesh_heading.find('DescriptorName'))
                qualifiers = [clean_text(qualifier) for qualifier in mesh_heading.findall('QualifierName')]
                mesh_terms.append(f"{descriptor} / {', '.join(qualifiers)}" if qualifiers else descriptor)

        keyword_list = pubmed_article.find('MedlineCitation/KeywordList') if pubmed_article is not None else None
        keywords = [clean_text(keyword) for keyword in keyword_list.findall('Keyword')] if keyword_list is not None else []
        if not mesh_terms and pmc_article is not None:
            mesh_group = pmc_article.find("front/article-meta/kwd-group[@kwd-group-type='MESH']")
            if mesh_group is not None:
                mesh_terms = [clean_text(keyword) for keyword in mesh_group.findall('kwd')]
        if not keywords and pmc_article is not None:
            for keyword_group in pmc_article.findall('front/article-meta/kwd-group'):
                if (keyword_group.get('kwd-group-type') or '').upper() != 'MESH':
                    keywords.extend(clean_text(keyword) for keyword in keyword_group.findall('kwd'))

        first_author_element = root.find(".//article-meta/contrib-group/contrib[@contrib-type='author']/name")
        if first_author_element is not None:
            given_names = clean_text(first_author_element.find('given-names'))
            surname = clean_text(first_author_element.find('surname'))
        else:
            pmc_author = pmc_article.find("front/article-meta/contrib-group/contrib[@contrib-type='author']/name") if pmc_article is not None else None
            pubmed_author = pubmed_article.find('MedlineCitation/Article/AuthorList/Author') if pubmed_article is not None else None
            given_names = clean_text(pmc_author.find('given-names')) if pmc_author is not None else clean_text(pubmed_author.find('ForeName')) if pubmed_author is not None else None
            surname = clean_text(pmc_author.find('surname')) if pmc_author is not None else clean_text(pubmed_author.find('LastName')) if pubmed_author is not None else None
        parsed_papers.append({
            'OriginalPaperDOI': doi, 'OriginalPaperPubMedID': pmid, 'PMCID': pmcid, 'XMLLINK': xml_link,
            'journal_nlm_ta': clean_text(root.find(".//journal-id[@journal-id-type='nlm-ta']")),
            'article_type': article.get('article-type'),
            'title': clean_text(root.find('.//article-meta/title-group/article-title')),
            'publication_year': clean_text(root.find('.//article-meta/pub-date/year')),
            'keywords': ' | '.join(term for term in keywords if term),
            'mesh_terms': ' | '.join(term for term in mesh_terms if term),
            'first_author_given_names': given_names or None,
            'first_author_surname': surname or None,
        })

    output_file = data_dir / 'fraudulent_papers_more_metadata.csv'
    pd.DataFrame(parsed_papers).to_csv(output_file, index=False)
    print(f'Saved {len(parsed_papers)} papers to {output_file}')
    print(f'Removed {removed_papers} non-research-article papers')
    print('Failed XML files:', failed_files)

if __name__ == '__main__':
    original_df = pd.read_csv('retraction_watch.csv') # NOTE: updated daily-ish: https://gitlab.com/crossref/retraction-watch-data/-/blob/main/retraction_watch.csv 
    filtered_df = filter_retractions(original_df)
    
    papers = find_pmcid(filtered_df)
    papers = check_pmcid_versions(papers)
    papers = get_xml_link(papers)

    os.mkdir('data') if not os.path.exists('data') else None
    data_dir = Path('data')
    os.mkdir('data/fraud-papers') if not os.path.exists('data/fraud-papers') else None
    papers_dir = Path('data/fraud-papers')
    download_papers(papers, papers_dir)
    get_more_metadata(papers, papers_dir, data_dir)
