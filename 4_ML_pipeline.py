import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# preprocessing with the embeds or word vectors or something
# pull the text from the features.csv, run the embeddings, then save them somewhere to be merged with the stuff below later

everything_metadata = pd.read_csv('data/everything_metadata.csv')
features = pd.read_csv('data/features.csv')

drop_columns = ['perplexity', 'per_word_perplexity'] # TODO: get rid of other columns that are not useful for classification
features = features.drop(columns=drop_columns)
id_columns = ['PMCID', 'pair_id', 'isFraud']
feature_list = [column for column in features.select_dtypes(include='number').columns if column not in id_columns]

# TODO: split the data in a good way, im just using 20 papers for for the sake of compute power
eligible_pair_ids = features.groupby('pair_id')['isFraud'].nunique()
selected_pair_ids = random.sample(eligible_pair_ids[eligible_pair_ids == 2].index.tolist(), 10)
selected_papers = features[features['pair_id'].isin(selected_pair_ids)]
selected_papers = selected_papers.groupby(['pair_id', 'isFraud'], as_index=False).head(1)
train_data, test_data = train_test_split(selected_papers, test_size=0.2, stratify=selected_papers['isFraud'], random_state=0)

# extract the features as the X, True/False as the Y
X_train = train_data[feature_list].copy()
Y_train = train_data['isFraud'].copy().astype(int)
X_test = test_data[feature_list].copy()
Y_test = test_data['isFraud'].copy().astype(int)

# all the models
models = {
    'logistic_regression': {
        'classifier': LogisticRegression(max_iter=2000, random_state=0),
        'use_scaler': True,
    },

    'linear_svm': {
        'classifier': LinearSVC(random_state=0),
        'use_scaler': True,
    },

    'rbf_svm': {
        'classifier': SVC(kernel='rbf', probability=True, random_state=0),
        'use_scaler': True,
    },

    'random_forest': {
        'classifier': RandomForestClassifier(n_estimators=300, random_state=0),
        'use_scaler': False,
    },
}

# pipeline: imputer (if needed to fill NaN), scaler, classifier
pipeline = Pipeline([
    ('scaler', StandardScaler() if models['logistic_regression']['use_scaler'] else 'passthrough'),
    ('classifier', models['logistic_regression']['classifier']),
])

# cross validate
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision': 'precision',
    'recall': 'recall',
}

results = cross_validate(pipeline, X_train, Y_train, scoring=scoring, n_jobs=-1, error_score='raise')
pipeline.fit(X_train, Y_train)
actual_results = pipeline.predict(X_test)

# save results and the actual trained model
