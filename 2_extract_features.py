import pandas as pd
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

fraud_papers_dir = Path('data/fraud-papers')
control_papers_dir = Path('data/control-papers')
fraud_metadata = pd.read_csv(Path('data/fraudulent_papers_metadata.csv'))
control_metadata = pd.read_csv(Path('data/control_papers_metadata.csv'))
fraud_metadata['isFraud'] = True
control_metadata['isFraud'] = False

def clean_text(element):
    if element is None:
        return None
    return ' '.join(''.join(element.itertext()).split())

big_metadata = []

for paper in tqdm(pd.concat([fraud_metadata[['PMCID', 'isFraud']], control_metadata[['PMCID', 'isFraud']]]).itertuples(), total=len(fraud_metadata) + len(control_metadata), unit=' papers'):
    if paper.isFraud:
        paper_file = fraud_papers_dir / f'{paper.PMCID}.xml'
    else:
        paper_file = control_papers_dir / f'{paper.PMCID}.xml'
    root = ET.parse(paper_file).getroot()
    body = root.find('.//body')
    citations = 0
    citations += len(body.findall('.//xref[@ref-type="bibr"]'))
    citations += len(body.findall('.//xref[@ref-type="ref"]'))
    citations += len(body.findall('.//xref[@ref-type="bib"]'))
    
    words = len(clean_text(body).split())

    big_metadata.append((paper.PMCID, paper.isFraud, (citations / words * 1000)))

big_metadata = pd.DataFrame(big_metadata, columns=['PMCID', 'isFraud', 'citation_density'])

non_fraud = big_metadata.loc[big_metadata['isFraud'] == False, 'citation_density'].dropna()
fraud = big_metadata.loc[big_metadata['isFraud'] == True, 'citation_density'].dropna()
statistic, p_value = mannwhitneyu(non_fraud, fraud, alternative='two-sided')
print(f'Mann-Whitney U test: U={statistic:.2f}, p={p_value:.4g}')

groups = [
    big_metadata.loc[big_metadata['isFraud'] == False, 'citation_density'].dropna().to_numpy(),
    big_metadata.loc[big_metadata['isFraud'] == True, 'citation_density'].dropna().to_numpy(),
]

plt.boxplot(
    [non_fraud, fraud],
    tick_labels=['Control Papers', 'Fraudulent Papers'],
    showfliers=False,
)

for x, values in enumerate(groups, start=1):
    jitter = np.random.normal(x, 0.04, size=len(values))
    plt.scatter(jitter, values, alpha=0.25, s=10)

plt.ylabel('Citations per 1000 words')
plt.title(f'Citation Density by Paper Type\np={p_value:.4g}')

plt.tight_layout()
plt.savefig('data/citation_density_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

big_metadata.to_csv('data/features.csv', index=False)