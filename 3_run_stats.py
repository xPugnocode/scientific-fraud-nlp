import os
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings('ignore', category=RuntimeWarning)

data = pd.read_csv('data/features.csv')

id_columns = ['PMCID', 'pair_id', 'isFraud']

feature_columns = [column for column in data.select_dtypes(include='number').columns if column not in id_columns]


results = []

def distribution_summary(values, prefix):
    values = values[np.isfinite(values)]

    summary = stats.describe(values, bias=False)
    q1, median, q3 = stats.scoreatpercentile(values, (25, 50, 75))
    spread_iqr = stats.iqr(values)
    lower_fence, upper_fence = q1 - 1.5 * spread_iqr, q3 + 1.5 * spread_iqr

    return {
        f'{prefix}_mean': summary.mean,
        f'{prefix}_median': median,
        f'{prefix}_std': np.sqrt(summary.variance),
        f'{prefix}_q1': q1,
        f'{prefix}_q3': q3,
        f'{prefix}_iqr': spread_iqr,
        f'{prefix}_min': summary.minmax[0],
        f'{prefix}_max': summary.minmax[1],
        f'{prefix}_range': np.ptp(values),
        f'{prefix}_skewness': summary.skewness,
        f'{prefix}_kurtosis': summary.kurtosis,
        f'{prefix}_n_outliers': np.count_nonzero((values < lower_fence) | (values > upper_fence)),
    }


def point_biserial_summary(feature_values, labels):
    valid = np.isfinite(feature_values) & np.isfinite(labels)
    feature_values = feature_values[valid]
    labels = labels[valid]

    test_result = stats.pointbiserialr(labels, feature_values)
    return test_result.statistic


for feature in feature_columns:
    paired = data.pivot(index='pair_id', columns='isFraud', values=feature)

    control = paired[False].to_numpy()
    retracted = paired[True].to_numpy()
    feature_values = data[feature].to_numpy()
    labels = data['isFraud'].astype(int).to_numpy()

    # positive difference means higher in retracted papers
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

    difference_summary = stats.describe(differences, bias=False) if differences.size else None

    result = {
        'feature': feature,
        'mean_difference': difference_summary.mean if difference_summary else np.nan,
        'median_difference': stats.scoreatpercentile(differences, 50) if differences.size else np.nan,
        'std_difference': np.sqrt(difference_summary.variance) if difference_summary else np.nan,
        'wilcoxon_statistic': statistic,
        'p_value': p_value,
        'rank_biserial': effect_size,
        'point_biserial_r': point_biserial_summary(feature_values, labels)
    }
    result.update(distribution_summary(control, 'control'))
    result.update(distribution_summary(retracted, 'retracted'))
    results.append(result)


results = pd.DataFrame(results)

# benjamini hochberg correction
results['q_value'] = multipletests(results['p_value'], alpha=0.05, method='fdr_bh')[1]

results['significant'] = results['q_value'] < 0.05
results = results.sort_values(['q_value'])

results.to_csv('data/feature_stats.csv', index=False)

things_to_plot = ['pos_prop_ADJ', 'duplicate_ngram_chr_fraction_5', 'passive_voice_rate']
fig, axes = plt.subplots(1, len(things_to_plot), figsize=(15, 4))
for axis, feature in zip(axes, things_to_plot):
    for is_fraud, label, color in ((False, 'Control', 'tab:blue'), (True, 'Retracted', 'tab:orange')):
        values = data.loc[data['isFraud'] == is_fraud, feature].dropna()
        axis.hist(values, bins=30, alpha=0.55, density=True, label=label, color=color)
    axis.set_title(feature)
    axis.set_xlabel('Value')
    axis.set_ylabel('Papers')
    axis.legend()

fig.tight_layout()
fig.savefig('data/selected_feature_distributions.png', dpi=150)
plt.close(fig)
