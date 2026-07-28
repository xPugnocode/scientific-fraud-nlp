import sklearn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import pandas as pd
import random
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import os

embeddings = pd.read_csv('data/pubmedbert_embeddings.csv')
# embeddings = None
if embeddings is not None:
    embedding_columns = [column for column in embeddings.columns if column != 'PMCID']

everything_metadata = pd.read_csv('data/everything_metadata.csv')
features = pd.read_csv('data/features.csv')

id_columns = ['PMCID', 'pair_id', 'isFraud']
drop_columns = ['perplexity', 'per_word_perplexity', 'proportion_ellipsis', 'proportion_bullet_points', 'contains_lorem ipsum', 'duplicate_line_chr_fraction', 'duplicate_paragraph_chr_fraction', 'n_acronym_mentions', 'n_citations']
features = features.drop(columns=drop_columns)
if embeddings is not None:
    features = features.merge(embeddings, on='PMCID', how='left')
feature_list = [column for column in features.select_dtypes(include='number').columns if column not in id_columns]
important_metadata_columns = ['PMCID', 'journal_nlm_ta', 'publication_year']

# NOTE: old code for choosing a smaller set of papers
# eligible_pair_ids = features.groupby('pair_id')['isFraud'].nunique()
# selected_pair_ids = random.sample(eligible_pair_ids[eligible_pair_ids == 2].index.tolist(), 50)
# selected_papers = features[features['pair_id'].isin(selected_pair_ids)]
# selected_papers = selected_papers.groupby(['pair_id', 'isFraud'], as_index=False).head(1)
selected_papers = features.groupby(['pair_id', 'isFraud'], as_index=False).head(1).merge(everything_metadata[important_metadata_columns], on='PMCID', how='left')

tested_splits = []

# testing many random group splits to pick the best one based on the similarity between the training and testing sets
for i in range(100):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
    train_index, test_index = next(splitter.split(selected_papers, groups=selected_papers['pair_id']))
    train_data = selected_papers.iloc[train_index]
    test_data = selected_papers.iloc[test_index]
    year_mean_diff = abs(train_data['publication_year'].mean() - test_data['publication_year'].mean())
    year_std_diff = abs(train_data['publication_year'].std() - test_data['publication_year'].std())
    journal_diff = train_data['journal_nlm_ta'].value_counts(normalize=True).sub(test_data['journal_nlm_ta'].value_counts(normalize=True), fill_value=0).abs().sum()
    tested_splits.append((i, year_mean_diff, year_std_diff, journal_diff))

tested_splits = pd.DataFrame(tested_splits, columns=['random_state', 'year_mean_diff', 'year_std_diff', 'journal_diff'])
tested_splits['score'] = tested_splits[['year_mean_diff', 'year_std_diff', 'journal_diff']].rank().sum(axis=1)
optimal_split = int(tested_splits.sort_values('score').iloc[0]['random_state'])
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=optimal_split)
train_index, test_index = next(splitter.split(selected_papers, groups=selected_papers['pair_id']))
train_data = selected_papers.iloc[train_index]
test_data = selected_papers.iloc[test_index]

# feature_list = embedding_columns

X_train = train_data[feature_list].copy()
Y_train = train_data['isFraud'].copy().astype(int)
print(X_train.shape, Y_train.shape)
print(X_train.dtypes.value_counts())
X_test = test_data[feature_list].copy()
Y_test = test_data['isFraud'].copy().astype(int)
print(X_test.shape, Y_test.shape)

# all the models

models = { 
    'logistic_regression': LogisticRegression(max_iter=2000, random_state=0), # NOTE: works fine for just the features, but not with the embeddings
    'linear_svm': LinearSVC(random_state=0),
    'rbf_svm': SVC(kernel='rbf', probability=True, random_state=0),
    'random_forest': RandomForestClassifier(n_estimators=300, random_state=0)
}

tested_models = []
OUTPUT_NAME = 'combined'
os.mkdir('models') if not os.path.exists('models') else None

for model in models.keys():
    # pipeline: imputer (if needed to fill NaN), scaler, classifier
    model_name = model
    # model_name = 'logistic_regression'
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', models[model_name]),
    ])

    # # cross validate
    # scoring = {
    #     'accuracy': 'accuracy',
    #     'balanced_accuracy': 'balanced_accuracy',
    #     'precision': 'precision',
    #     'recall': 'recall',
    # }

    # results = cross_validate(pipeline, X_train, Y_train, scoring=scoring, n_jobs=-1, error_score='raise')
    pipeline.fit(X_train, Y_train)
    actual_results = pipeline.predict(X_test)
    if hasattr(pipeline, 'predict_proba'):
        actual_scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        actual_scores = pipeline.decision_function(X_test)

    test_results = test_data.copy()
    test_results['predicted_isFraud'] = actual_results
    test_results['predicted_score_isFraud'] = actual_scores
    # test_results.to_csv(f'data/{OUTPUT_NAME}_results.csv', index=False)

    tn, fp, fn, tp = confusion_matrix(Y_test, actual_results, labels=[0, 1]).ravel()
    test_count = len(Y_test)
    metrics = {
        'model_name': model_name,
        'true_positive_pct': 100 * tp / test_count,
        'true_negative_pct': 100 * tn / test_count,
        'false_positive_pct': 100 * fp / test_count,
        'false_negative_pct': 100 * fn / test_count,
        'total_accuracy': 100 * (tp + tn) / test_count,
        'precision': 100 * precision_score(Y_test, actual_results, zero_division=0),
        'recall': 100 * recall_score(Y_test, actual_results, zero_division=0),
        'f1': 100 * f1_score(Y_test, actual_results, zero_division=0),
        'roc_auc': 100 * roc_auc_score(Y_test, actual_scores),
    }
    tested_models.append(metrics)

    dump(pipeline, f'models/{OUTPUT_NAME}_trained_{model_name}_model.joblib')

pd.DataFrame(tested_models).to_csv(f'data/{OUTPUT_NAME}_metrics.csv', index=False)
