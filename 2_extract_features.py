import pandas as pd
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import warnings
import spacy
import textdescriptives as td
from textacy.text_stats import diversity
from textacy.extract.acros import acronyms, acronyms_and_definitions
from spacytextblob.spacytextblob import SpacyTextBlob

from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=InconsistentVersionWarning, module='sklearn')

columns = [
    'PMCID',
    'pair_id',
    'isFraud',
]

nlp = spacy.load('en_core_sci_md')
nlp.add_pipe('textdescriptives/all')
nlp.add_pipe('spacytextblob')

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

def get_features(body):
    article = clean_text(body)
    doc = nlp(article) # TODO: could use nlp.pipe to do batching which would (hopefully) speed things up

    features = td.extract_dict(doc, include_text=False)[0]
    
    words = [token for token in doc if token.is_alpha]
    sentences = list(doc.sents)

    more_features = []
    more_features.append(('segmented_ttr', diversity.segmented_ttr(doc, variant='moving-avg')))
    more_features.append(('mltd', diversity.mtld(doc)))
    more_features.append(('hdd', diversity.hdd(doc)))

    modal_verbs = [token for token in doc if token.tag_ == 'MD']
    passive_sentences = [sentence for sentence in sentences if any(token.dep_ in {'nsubjpass', 'auxpass'} for token in sentence)]
    negations = [token for token in doc if token.dep_ == 'neg']
    first_person_singular_pronouns = [token for token in doc if token.lower_ in {'i', 'me', 'my'}]
    first_person_plural_pronouns = [token for token in doc if token.lower_ in {'we', 'us', 'our'}]
    numbers = [token for token in doc if token.like_num]

    more_features.append(('modal_verb_rate', len(modal_verbs) / len(words) * 1000 if words else 0.0))
    more_features.append(('passive_voice_rate', len(passive_sentences) / len(sentences) if sentences else 0.0))
    more_features.append(('negation_rate', len(negations) / len(words) * 1000 if words else 0.0))
    more_features.append(('first_person_singular_rate', len(first_person_singular_pronouns) / len(words) * 1000 if words else 0.0))
    more_features.append(('first_person_plural_rate', len(first_person_plural_pronouns) / len(words) * 1000 if words else 0.0))
    more_features.append(('number_rate', len(numbers) / len(words) * 1000 if words else 0.0))

    acronym_mentions = [token.text.upper() for token in acronyms(doc)]
    acronym_definitions = acronyms_and_definitions(doc)
    more_features.append(('n_acronym_mentions', len(acronym_mentions)))
    more_features.append(('n_unique_acronyms', len(set(acronym_mentions))))
    more_features.append(('n_defined_acronyms', sum(bool(definition) for definition in acronym_definitions.values())))
    more_features.append(('acronym_density', len(acronym_mentions) / len(words) * 1000))
    more_features.append(('ratio_defined_acronyms', sum(bool(definition) for definition in acronym_definitions.values()) / len(set(acronym_mentions)) if len(set(acronym_mentions)) > 0 else 0.0))

    citations = len(body.findall('.//xref[@ref-type="bibr"]'))
    citations += len(body.findall('.//xref[@ref-type="ref"]'))
    citations += len(body.findall('.//xref[@ref-type="bib"]'))
    more_features.append(('n_citations', citations))
    more_features.append(('citation_density', citations / len(words) * 1000))

    more_features.append(('polarity', doc._.blob.polarity))
    more_features.append(('subjectivity', doc._.blob.subjectivity))

    more_features.append(('n_words', len(words)))
    
    # NOTE: i will need to narrow down this list a bit
    features.update(dict(more_features))
    return features

for paper in tqdm(pd.concat([fraud_metadata[['PMCID', 'pair_id', 'isFraud']], control_metadata[['PMCID', 'pair_id', 'isFraud']]]).itertuples(), total=len(fraud_metadata) + len(control_metadata), unit=' papers'):
    if paper.isFraud:
        paper_file = fraud_papers_dir / f'{paper.PMCID}.xml'
    else:
        paper_file = control_papers_dir / f'{paper.PMCID}.xml'
    
    root = ET.parse(paper_file).getroot()
    body = root.find('.//body')
    features = get_features(body)

    big_metadata.append((paper.PMCID, paper.pair_id, paper.isFraud, *features.values()))

big_metadata = pd.DataFrame(big_metadata, columns=(columns + list(features.keys())))
paper_metadata = pd.concat([fraud_metadata, control_metadata], ignore_index=True)
merged_metadata = pd.merge(big_metadata, paper_metadata, on=['PMCID', 'pair_id'], how='left')

big_metadata.to_csv('data/features.csv', index=False)
merged_metadata.to_csv('data/merged_metadata.csv', index=False)
