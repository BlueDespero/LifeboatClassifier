import pandas as pd

from tests.utils import load_results
from tests.knn.k_crossvalidation import encoders

results = load_results("../../Classification_algorithms/K_Nearest_Neighbors/predictions")
reverse_encoders = {v: k for k, v in encoders.items()}

results['encoder'] = results['encoder'].apply(lambda x: reverse_encoders[x])


best_all = None
best_all_err = 1.5
best_results = []

for _, group in results.groupby(['encoder', 'distance'], as_index=False):
    best_group = group.iloc[group['err_rate'].idxmin()]
    best_results.append(best_group)
    if best_all_err > best_group['err_rate']:
        best_all = best_group
        best_all_err = best_group['err_rate']

print("Best off all the results:")
print(best_all)

summary = pd.concat(best_results, axis=1).T
summary = summary[['encoder', 'distance', 'k', 'err_rate', 'standard_deviation']]
summary.to_excel('../../Classification_algorithms/K_Nearest_Neighbors/predictions/knn_crossval_summary.xlsx', index=False)
print(summary)

