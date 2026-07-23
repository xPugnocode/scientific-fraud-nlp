from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from tqdm import tqdm


MODEL_NAME = 'neuml/pubmedbert-base-embeddings-8M'

XML_ROOTS = (Path('data/control-papers'), Path('data/fraud-papers'))

static = StaticEmbedding.from_model2vec('neuml/pubmedbert-base-embeddings-8M')
model = SentenceTransformer(modules=[static])

# model = SentenceTransformer('neuml/biomedbert-small-embeddings') # TODO: run this later tonight I anticipate ~2 hours of runtime

sentence_nlp = spacy.blank('en')
sentence_nlp.add_pipe('sentencizer')

def clean_text(element):
    if element is None:
        return None
    return ' '.join(''.join(element.itertext()).split())    


def create_paper_vector(text):
    doc = sentence_nlp(text)
    sentences = [sentence.text.strip() for sentence in doc.sents]

    sentence_vectors = []

    for start in range(0, len(sentences), 16):
        batch = sentences[start:start + 16]
        vectors = model.encode(batch, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
        sentence_vectors.append(vectors)

    sentence_vectors = np.vstack(sentence_vectors)

    return sentence_vectors.mean(axis=0)


features = pd.read_csv('data/features.csv')

xml_paths = {}

for path in Path('data/fraud-papers').glob('*.xml'):
    xml_paths[path.stem] = path

for path in Path('data/control-papers').glob('*.xml'):
    xml_paths[path.stem] = path

embedding_rows = []

for pmcid in tqdm(features['PMCID'].astype(str).unique()):
    xml_path = xml_paths.get(pmcid)

    root = ET.parse(xml_path).getroot()
    body = root.find('.//body')
    body = clean_text(body)
    paper_vector = create_paper_vector(body)

    row = {'PMCID': pmcid}
    for index, value in enumerate(paper_vector):
        row[f'pubmedbert_{index}'] = value

    embedding_rows.append(row)

embeddings = pd.DataFrame(embedding_rows)
embeddings.to_csv('data/pubmedbert_embeddings.csv', index=False)
print(embeddings.shape)
