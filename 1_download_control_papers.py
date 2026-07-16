import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib3.util.retry import Retry
from scispacy.linking import EntityLinker
import spacy
import re
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=InconsistentVersionWarning, module='sklearn')


nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe('scispacy_linker', config={'linker_name': 'mesh', 'threshold': 0.8, 'max_entities_per_mention': 1})
linker = nlp.get_pipe('scispacy_linker')


ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
IDCONV_URL = 'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/'
S3_BASE = 'https://pmc-oa-opendata.s3.amazonaws.com'
load_dotenv() 
API_KEY = os.getenv('ENTREZ_API')

session = requests.Session()
retry = Retry(status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET"), backoff_factor=2)
adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
session.mount("https://", adapter)

data_dir = Path('data')
papers_dir = data_dir / 'fraud-papers'
control_papers_dir = data_dir / 'control-papers'
fraud_metadata_df = pd.read_csv(data_dir / 'fraudulent_papers_metadata.csv')

def add_to_query(query, things, connector='AND'):
    if not isinstance(things, list):
        things = [things]
    if len(things) == 1:
        query += f' {connector} {things[0]}'
    else:
        query += f' {connector} ({' OR '.join(things)})'
    return query

def clean_text(element):
    if element is None:
        return None
    return ' '.join(''.join(element.itertext()).split())


RETRACTION_TITLE_PREFIX = re.compile(r'^RETRACTED(?:\s+ARTICLE|\s+AND\s+REMOVED)?\s*:\s*', re.IGNORECASE)


def clean_title(element):
    title = clean_text(element)
    return RETRACTION_TITLE_PREFIX.sub('', title)


def get_important_title_words(title):
    # return []
    doc = nlp(title)
    terms = []

    for entity in doc.ents:
        if not entity._.kb_ents:
            continue

        mesh_id = entity._.kb_ents[0][0]
        mesh_entity = linker.kb.cui_to_entity[mesh_id]

        terms.append(mesh_entity.canonical_name)
    # print(terms)
    return terms


def split_terms(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [term.strip() for term in str(value).split('|') if term.strip()]


def build_control_query(row, year_window=1, include_journal=True, topic_field='mh'):
    year_range = f'{int(row.publication_year) - year_window}:{int(row.publication_year) + year_window}[dp]'
    if include_journal:
        query = f'"{row.journal_nlm_ta}"[ta]'
        query = add_to_query(query, year_range)
    else:
        query = year_range
    query = add_to_query(query, 'pubmed pmc[sb]')
    query = add_to_query(query, 'journal article[pt]')

    excluded_types = ['retracted publication[pt]', 'retraction of publication[pt]','expression of concern[pt]', 'review[pt]', 'systematic review[pt]','meta-analysis[pt]', 'editorial[pt]', 'letter[pt]', 'comment[pt]', 'news[pt]', 'case reports[pt]', 'published erratum[pt]', 'preprint[pt]']
    excluded_statuses = ['hasretractionin', 'hasexpressionofconcernin', 'hasretractedandrepublishedin']
    excluded_terms = excluded_types + excluded_statuses
    query = add_to_query(query, excluded_terms, connector='NOT')
    query = add_to_query(query, f'{int(row.OriginalPaperPubMedID)}[pmid]', connector='NOT')
    if pd.notna(row.first_author_surname) and row.first_author_surname:
        author_name = str(row.first_author_surname)
        if pd.notna(row.first_author_given_names) and row.first_author_given_names:
            author_name += f' {row.first_author_given_names}'
        author_name = author_name.replace('.', '')
        query = add_to_query(query, f'{author_name}[au]', connector='NOT')

    mesh_terms = [term.split(' / ', 1)[0] for term in split_terms(row.mesh_terms)]
    topic_terms = [f'"{term}"[{topic_field}]' for term in mesh_terms] if topic_field else []
    if topic_terms:
        query = add_to_query(query, topic_terms)
    return query


def search_pubmed(query, retmax=20):
    params = {'db': 'pubmed', 'term': query, 'retmode': 'json', 'retmax': retmax}
    if API_KEY:
        params['api_key'] = API_KEY
    r = session.get(ESEARCH_URL, params=params)
    r.raise_for_status()
    return r.json().get('esearchresult', {}).get('idlist', [])


def fetch_pubmed_metadata(pmid):
    params = {'db': 'pubmed', 'id': pmid}
    if API_KEY:
        params['api_key'] = API_KEY
    r = session.get(EFETCH_URL, params=params)
    r.raise_for_status()

    pubmed_article = ET.fromstring(r.content).find('PubmedArticle')
    if pubmed_article is None:
        return None
    citation = pubmed_article.find('MedlineCitation')
    mesh_headings = pubmed_article.find('MedlineCitation/MeshHeadingList')
    keyword_lists = pubmed_article.findall('MedlineCitation/KeywordList')
    pubmed_author = pubmed_article.find('MedlineCitation/Article/AuthorList/Author') if pubmed_article is not None else None
    given_names = clean_text(pubmed_author.find('ForeName')) if pubmed_author is not None else None
    surname = clean_text(pubmed_author.find('LastName')) if pubmed_author is not None else None
    return {
        'PMID': clean_text(citation.find('PMID')),
        'title': clean_text(citation.find('Article/ArticleTitle')),
        'first_author_given_names': given_names or None,
        'first_author_surname': surname or None,
        'publication_types': [clean_text(item) for item in citation.findall('Article/PublicationTypeList/PublicationType')],
        'mesh_terms': [clean_text(item.find('DescriptorName')) for item in mesh_headings.findall('MeshHeading')] if mesh_headings is not None else [],
        'keywords': [clean_text(keyword) for keyword_list in keyword_lists for keyword in keyword_list.findall('Keyword')],
    }

def get_valid_control_xml(pmcid):
    r = session.get(S3_BASE, params={
        'list-type': '2',
        'prefix': f'{pmcid}.',
        'delimiter': '/',
    })
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    versions = [
        item.text.rstrip('/')
        for item in root.findall('s3:CommonPrefixes/s3:Prefix', namespaces=ns)
    ]
    for version in versions:
        for metadata_url in (
            f'{S3_BASE}/metadata/{version}.json',
            f'{S3_BASE}/{version}/{version}.json',
        ):
            metadata = session.get(metadata_url)
            if not metadata.ok:
                continue
            xml_url = metadata.json().get('xml_url')
            if not xml_url:
                break
            xml_response = session.get(xml_url.replace('s3://pmc-oa-opendata', S3_BASE))
            if not xml_response.ok:
                continue
            try:
                root = ET.fromstring(xml_response.content)
            except ET.ParseError:
                continue
            article = root.find('article') if root.tag == 'pmc-articleset' else root
            if article is not None and article.get('article-type') == 'research-article':
                return str(xml_url), xml_response.content
    return None


def choose_control(row, claimed_pmids=None, claimed_pmids_lock=None):
    seen_pmids = set()
    mesh_terms = [term.split(' / ', 1)[0] for term in split_terms(row.mesh_terms)]
    search_fallbacks = (
        (('mh', True, 1), ('mh', True, 2), ('mh', False, 2), ('tiab', False, 2), (None, False, 2))
        if mesh_terms else ((None, True, 1), (None, True, 2), (None, False, 2))
    )
    for topic_field, include_journal, year_window in search_fallbacks:
        query = build_control_query(
            row,
            year_window=year_window,
            include_journal=include_journal,
            topic_field=topic_field,
        )
        candidate_pmids = search_pubmed(query, retmax=20)
        new_pmids = [pmid for pmid in candidate_pmids if pmid not in seen_pmids]
        if not new_pmids:
            continue
        seen_pmids.update(new_pmids)

        r = session.get(IDCONV_URL, params={'idtype': 'pmid', 'ids': ','.join(new_pmids), 'format': 'json'})
        r.raise_for_status()
        pmcids = {
            str(record.get('pmid')): record.get('pmcid')
            for record in r.json().get('records', [])
            if record.get('pmcid')
        }

        for pmid in new_pmids:
            pmcid = pmcids.get(pmid)
            if not pmcid:
                continue
            if claimed_pmids is not None:
                with claimed_pmids_lock:
                    if pmid in claimed_pmids:
                        continue
            valid_xml = get_valid_control_xml(pmcid)
            if valid_xml is None:
                continue
            if claimed_pmids is not None:
                with claimed_pmids_lock:
                    if pmid in claimed_pmids:
                        continue
                    claimed_pmids.add(pmid)
            xml_url, xml_content = valid_xml
            return (
                str(int(row.OriginalPaperPubMedID)),
                str(pmid),
                str(pmcid),
                xml_url,
                xml_content,
            )
    # print(f'debug no valid control paper found for {row[0]}')
    return None


def find_pmcid(papers):
    results = []
    for start in tqdm(range(0, len(papers), 200), desc='Checking PMCID', unit=' batches'): 
        batch = papers[start:start + 200] # maximum limit of 200 papers per request
        # time.sleep(1/3)
        r = session.get(IDCONV_URL, params={'idtype': 'pmid', 'ids': ','.join(pmid for _, pmid in batch), 'format': 'json'})
        r.raise_for_status()
        data = r.json()

        records = {str(record.get('pmid')): record for record in data.get('records', [])}
        for fraud_pmid, pmid in batch:
            pmcid = records.get(pmid, {}).get('pmcid')
            if pmcid is not None:
                results.append((fraud_pmid, pmid, pmcid))    
    # print(results)
    # print('Total papers with PMCID:', len(results))
    return results


def check_pmcid_versions(papers):
    def get_versions(paper):
        fraud_pmid, pmid, pmcid = paper
        r = session.get(S3_BASE, params={'list-type': '2', 'prefix': f'{pmcid}.', 'delimiter': '/'})
        r.raise_for_status()
        # print(r.content)
        root = ET.fromstring(r.text)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        versions = [p.text for p in root.findall('s3:CommonPrefixes/s3:Prefix', namespaces=ns)]
        if not versions:
            return None
        versions = [version.rstrip('/') for version in versions]

        return fraud_pmid, pmid, versions

    # each request is independent, so fastest way to speed it up is doing it concurrently

    with ThreadPoolExecutor(max_workers=20) as executor:
        checked_results = tqdm(executor.map(get_versions, papers), total=len(papers), desc='Checking PMCID versions', unit=' papers')
        papers = []
        seen_fraud_pmids = set()
        for result in checked_results:
            if result is None:
                continue
            fraud_pmid = result[0]
            if fraud_pmid in seen_fraud_pmids:
                continue
            seen_fraud_pmids.add(fraud_pmid)
            papers.append(result)
        # print('Total papers with downloadable versions:', len(new_results))
    return papers


def get_xml_link(papers):
    def check_xml_availability(paper):
        fraud_pmid, pmid, versions = paper
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
                return str(fraud_pmid), str(pmid), str(pmcid), str(xml_url)
        # print(f'No XML found for PMID: {pmid}')
        return fraud_pmid, pmid, versions[0].rsplit('.', 1)[0], None

    with ThreadPoolExecutor(max_workers=20) as executor:
        availability = list(tqdm(executor.map(check_xml_availability, papers), total=len(papers), desc='Fetching XML links', unit=' papers'))
    # print(len(availability), availability)
    return pd.DataFrame(availability, columns=['FraudPaperPubMedID', 'ControlPaperPubMedID', 'PMCID', 'XMLLink']).dropna()


def download_papers(papers, papers_dir):
    def download_paper(paper):
        _, _, pmcid, _, xml_content = paper
        xml_filename = papers_dir / f'{pmcid}.xml'
        xml_filename.write_bytes(xml_content)
        

    with ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(download_paper, papers.values), total=len(papers), desc='Downloading papers', unit=' papers'))


def save_control_metadata(control_papers, papers_dir, data_dir):
    pubmed_articles = {}

    pmids = [str(pmid) for pmid in control_papers['ControlPaperPubMedID']]
    for start in tqdm(range(0, len(pmids), 200), desc='Fetching PubMed metadata', unit=' batches'):
        r = session.get(EFETCH_URL, params={'db': 'pubmed', 'id': ','.join(pmids[start:start + 200])})
        r.raise_for_status()
        for pubmed_article in ET.fromstring(r.content).findall('PubmedArticle'):
            pubmed_articles[clean_text(pubmed_article.find('MedlineCitation/PMID'))] = pubmed_article

    parsed_papers = []
    failed_files = []
    removed_papers = 0

    for row in tqdm(control_papers.itertuples(index=False), total=len(control_papers), desc='Parsing XML', unit=' papers'):
        original_pmid, pmid, pmcid, xml_link = row
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
        first_author_element = root.find(".//article-meta/contrib-group/contrib[@contrib-type='author']/name")
        if first_author_element is not None:
            given_names = clean_text(first_author_element.find('given-names'))
            surname = clean_text(first_author_element.find('surname'))
        else:
            pubmed_author = pubmed_article.find('MedlineCitation/Article/AuthorList/Author') if pubmed_article is not None else None
            given_names = clean_text(pubmed_author.find('ForeName')) if pubmed_author is not None else None
            surname = clean_text(pubmed_author.find('LastName')) if pubmed_author is not None else None
        parsed_papers.append({
            'OriginalPaperPubMedID': original_pmid,
            'ControlPaperPubMedID': pmid,
            'PMCID': pmcid,
            'XMLLINK': xml_link,
            'journal_nlm_ta': clean_text(root.find(".//journal-id[@journal-id-type='nlm-ta']")),
            'article_type': article.get('article-type'),
            'title': clean_title(root.find('.//article-meta/title-group/article-title')),
            'publication_year': clean_text(root.find('.//article-meta/pub-date/year')),
            'first_author_given_names': given_names or None,
            'first_author_surname': surname or None,
        })

    output_file = data_dir / 'control_papers_metadata.csv'
    pd.DataFrame(parsed_papers).to_csv(output_file, index=False)
    print(f'Saved {len(parsed_papers)} papers to {output_file}')
    print(f'Removed {removed_papers} non-research-article papers')
    print('Failed XML files:', failed_files)


if __name__ == '__main__':
    claimed_control_pmids = set()
    claimed_control_pmids_lock = Lock()

    def find_control_paper(row):
        try:
            return choose_control(row, claimed_control_pmids, claimed_control_pmids_lock)
        except requests.RequestException as e:
            print(f'Failed finding control paper for PMID {row.OriginalPaperPubMedID}: {e}')
        return None

    rows = list(fraud_metadata_df.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=20) as executor:
        checked_results = list(tqdm(executor.map(find_control_paper, rows), total=len(rows), desc='Finding control papers', unit=' papers'))
        control_papers = pd.DataFrame(
            [result for result in checked_results if result is not None],
            columns=['OriginalPaperPubMedID', 'ControlPaperPubMedID', 'PMCID', 'XMLLink', 'XMLContent'],
        )
    os.mkdir('data/control-papers') if not os.path.exists('data/control-papers') else None
    download_papers(control_papers, control_papers_dir)
    control_papers = control_papers.drop(columns='XMLContent')
    save_control_metadata(control_papers, control_papers_dir, data_dir)
    xml_paths = [control_papers_dir / f'{pmcid}.xml' for pmcid in control_papers['PMCID']]
    # print('Saved XML paths:', xml_paths)
    # print('Full text found:', [xml_path.exists() for xml_path in xml_paths])
