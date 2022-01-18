from tests.utils import load_results
from tests.knn.k_crossvalidation import encoders

results = load_results("../../Classification_algorithms/K_Nearest_Neighbors/predictions")
reverse_encoders = {v: k for k, v in encoders.items()}
print(reverse_encoders)

results['encoder'] = results['encoder'].apply(lambda x: reverse_encoders[x])

print(results['encoder'])

best_all = None
best_all_err = 1.5

for _, group in results.groupby(['encoder', 'distance'], as_index=False):
    best_group = group.iloc[group['err_rate'].idxmin()]
    print(best_group)
    print("\n")
    if best_all_err > best_group['err_rate']:
        best_all = best_group
        best_all_err = best_group['err_rate']

print("Best off all the results:")
print(best_all)

