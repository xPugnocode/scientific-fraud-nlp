import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg

from statsmodels.stats.multitest import multipletests


data = pd.read_csv('data/features.csv')

id_columns = ['PMCID', 'pair_id', 'isFraud']

feature_columns = [column for column in data.select_dtypes(include='number').columns if column not in id_columns]


results = []

for feature in feature_columns:
    paired = data.pivot(index='pair_id', columns='isFraud', values=feature)

    control = paired[False].to_numpy()
    retracted = paired[True].to_numpy()

    # Positive difference means higher in retracted papers
    differences = retracted - control

    if np.all(differences == 0):
        statistic = 0.0
        p_value = 1.0
        effect_size = 0.0
    else:
        test_result = pg.wilcoxon(retracted, control)
        statistic = test_result.at['Wilcoxon', 'W_val']
        p_value = test_result.at['Wilcoxon', 'p_val']
        effect_size = test_result.at['Wilcoxon', 'RBC']

    results.append({'feature': feature, 'control_median': np.median(control), 'retracted_median': np.median(retracted), 'median_difference': np.median(differences), 'wilcoxon_statistic': statistic, 'p_value': p_value, 'rank_biserial': effect_size})


results = pd.DataFrame(results)

# Benjamini–Hochberg correction across all features
results['q_value'] = multipletests(results['p_value'], alpha=0.05, method='fdr_bh')[1]

results['significant'] = results['q_value'] < 0.05
results = results.sort_values(['q_value'])

results.to_csv('data/wilcoxon_feature_results.csv', index=False)
