import os
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

IDCONV_URL = 'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/' # https://pmc.ncbi.nlm.nih.gov/tools/id-converter-api/
S3_BASE = 'https://pmc-oa-opendata.s3.amazonaws.com' # https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/

def filter_retractions(big_df):
    small_df = big_df[(
        (big_df['Reason'].str.contains('Falsification') | big_df['Reason'].str.contains('Manipulation'))
        & (big_df['ArticleType'].str.contains('Research Article'))
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
        r = requests.get(IDCONV_URL, params={'idtype': 'pmid', 'ids': ','.join(pmid for _, pmid in batch), 'format': 'json'})
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
    def get_latest_version(paper):
        doi, pmid, pmcid = paper
        r = requests.get(S3_BASE, params={'list-type': '2', 'prefix': pmcid, 'delimiter': '/'})
        r.raise_for_status()
        # print(r.content)
        root = ET.fromstring(r.text)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        versions = [p.text for p in root.findall('s3:CommonPrefixes/s3:Prefix', namespaces=ns)]
        if not versions:
            return None
        version_numbers = [(str(v).split('.')[-1][:-1]) for v in versions]
        latest_version = pmcid + '.' + max(version_numbers)
        return doi, pmid, pmcid, latest_version

    # each request is independent, so fastest way to speed it up is doing it concurrently

    with ThreadPoolExecutor(max_workers=20) as executor:
        checked_results = tqdm(executor.map(get_latest_version, papers), total=len(papers), desc='Checking PMCID versions', unit=' papers')
        papers = [result for result in checked_results if result is not None]
        # print('Total papers with downloadable versions:', len(new_results))
    return papers



def get_xml_link(papers):
    def check_xml_availability(paper):
        doi, pmid, pmcid, version = paper
        r = requests.get(f'{S3_BASE}/metadata/{version}.json')
        try:
            r.raise_for_status()
        except requests.RequestException as e: # in the future these will need to be changed to just 404 errors so i can handle 429 and 503
            print(f'Error fetching metadata for {pmcid}: {e}')
            r = requests.get(f'{S3_BASE}/{version}/{version}.json') # super weird API with two different possible places for the metadata to be stored
            try:
                r.raise_for_status()
            except requests.RequestException as e:
                print(f'Error fetching metadata for {pmcid}: {e}')
                r = requests.get(f'{S3_BASE}/metadata/{pmcid}.1.json')
                try:
                    r.raise_for_status()
                except requests.RequestException as e:
                    print(f'Error fetching metadata for {pmcid}: {e}')
                    r = requests.get(f'{S3_BASE}/{pmcid}.1/{pmcid}.1.json')
                    try:
                        r.raise_for_status()
                    except requests.RequestException as e:
                        print(f'Error fetching metadata for {pmcid}: {e}')
                        return doi, pmid, pmcid, None
        meta = r.json()

        return str(doi), str(pmid),str(pmcid), str(meta.get('xml_url'))

    with ThreadPoolExecutor(max_workers=20) as executor:
        availability = list(tqdm(executor.map(check_xml_availability, papers), total=len(papers), desc='Fetching XML links', unit='papers'))
    # print(len(availability), availability)
    return pd.DataFrame(availability, columns=['OriginalPaperDOI', 'OriginalPaperPubMedID', 'PMCID', 'XMLLink']).dropna()

def download_papers(papers, papers_dir):
    def download_paper(paper):
        _, _, pmcid, xml_url = paper
        xml_url = xml_url.replace('s3://pmc-oa-opendata', S3_BASE)
        r = requests.get(xml_url)
        r.raise_for_status()
        xml_filename = papers_dir / f'{pmcid}.xml'
        xml_filename.write_bytes(r.content)
        
    with ThreadPoolExecutor(max_workers=20) as executor:
        tqdm(executor.map(download_paper, papers.values), total=len(papers), desc='Downloading papers', unit=' papers')

if __name__ == '__main__':
    original_df = pd.read_csv('retraction_watch.csv') # NOTE: updated daily-ish: https://gitlab.com/crossref/retraction-watch-data/-/blob/main/retraction_watch.csv 
    filtered_df = filter_retractions(original_df)
    
    papers = find_pmcid(filtered_df)
    papers = check_pmcid_versions(papers)
    papers = get_xml_link(papers)

    os.mkdir('data') if not os.path.exists('data') else None
    data_dir = Path('data')
    papers.to_csv(data_dir / 'fraudulent_papers_basic_metadata.csv', index=False)

    os.mkdir('data/fraud-papers') if not os.path.exists('data/fraud-papers') else None
    papers_dir = Path('data/fraud-papers')
    download_papers(papers, papers_dir)