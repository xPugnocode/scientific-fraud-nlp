import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import spacy
import textdescriptives as td
from joblib import load
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sklearn.exceptions import InconsistentVersionWarning
from spacytextblob.spacytextblob import SpacyTextBlob
from textacy.extract.acros import acronyms, acronyms_and_definitions
from textacy.text_stats import diversity


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=InconsistentVersionWarning, module='sklearn')

MODEL_NAME = 'neuml/pubmedbert-base-embeddings-8M'
MODEL_PATH = Path('models/combined_trained_rbf_svm_model.joblib')

pipeline = None
nlp = None
model = None
sentence_nlp = None


def load_everything():
    global pipeline, nlp, model, sentence_nlp

    pipeline = load(MODEL_PATH)

    nlp = spacy.load('en_core_sci_md')
    nlp.add_pipe('textdescriptives/all')
    nlp.add_pipe('spacytextblob')

    static = StaticEmbedding.from_model2vec(MODEL_NAME)
    model = SentenceTransformer(modules=[static])

    sentence_nlp = spacy.blank('en')
    sentence_nlp.add_pipe('sentencizer')


def clean_text(text):
    return ' '.join(text.split())


def get_features(article, citations):
    doc = nlp(article)
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
    more_features.append(('acronym_density', len(acronym_mentions) / len(words) * 1000 if words else 0.0))
    more_features.append(('ratio_defined_acronyms', sum(bool(definition) for definition in acronym_definitions.values()) / len(set(acronym_mentions)) if len(set(acronym_mentions)) > 0 else 0.0))

    more_features.append(('n_citations', citations))
    more_features.append(('citation_density', citations / len(words) * 1000 if words else 0.0))

    more_features.append(('polarity', doc._.blob.polarity))
    more_features.append(('subjectivity', doc._.blob.subjectivity))

    more_features.append(('n_words', len(words)))

    features.update(dict(more_features))
    return features


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


def predict_paper(text_file, citations):
    article = clean_text(text_file.read_text(encoding='utf-8'))
    features = get_features(article, citations)
    paper_vector = create_paper_vector(article)

    row = dict(features)
    for index, value in enumerate(paper_vector):
        row[f'pubmedbert_{index}'] = value

    feature_list = list(pipeline.feature_names_in_)
    missing_columns = [column for column in feature_list if column not in row]
    if missing_columns:
        raise ValueError(f'Missing model input columns: {missing_columns}')

    X = pd.DataFrame([{column: row[column] for column in feature_list}])

    prediction = int(pipeline.predict(X)[0])
    probability_control, probability_fraud = pipeline.predict_proba(X)[0]

    return {
        'text_file': str(text_file),
        'prediction': 'fraud' if prediction == 1 else 'legitimate',
        'predicted_isFraud': bool(prediction),
        'probability_control': float(probability_control),
        'probability_fraud': float(probability_fraud),
        'confidence': float(max(probability_control, probability_fraud)),
        'n_citations': citations,
        'citation_density': float(row['citation_density']),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('text_file', type=Path)
    parser.add_argument('citations', type=int)
    return parser.parse_args()


args = parse_args()
load_everything()
results = predict_paper(args.text_file, args.citations)
print(json.dumps(results, indent=2))
