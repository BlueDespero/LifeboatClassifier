from tests.utils import cross_validation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm
plt.style.use('seaborn-whitegrid')


from Classification_algorithms.Random_Forest.random_forest import RandomForest

train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')

train['Name'] = train['Name'].apply(lambda full_name: full_name.split(',')[0])
train.rename(columns={"Survived": "target"}, inplace=True)

train_x = train.iloc[:, 2:]
train_y = train.iloc[:, 1]

k = 15

r = [75,50,25,10,5,1][::-1]
# r = [2,1]
# print(r)
min_val = 1
args = []
err = []
std = []
criterions = ["infogain", "infogain_ratio", "mean_err_rate", "gini"]
# criterions = ["infogain"]
for criterion in tqdm(criterions,position=0,leave=False):
    error_rates = []
    standard_deviations = []
    for i in tqdm(r, position=1,leave=False):
        result = cross_validation(train_x, train_y, classifier=RandomForest, folds=k, criterion=criterion,
                                  nattrs=1, trees_no=i,verbose=False)
        error_rates.append(result['err_rate'])
        standard_deviations.append(result['standard_deviation'])

    standard_deviations = np.array(standard_deviations)
    error_rates = np.array(error_rates)

    upper_bound = error_rates+2*standard_deviations
    arg_index = np.argmin(upper_bound)
    if upper_bound[arg_index]<min_val:
        min_val = upper_bound[arg_index]
        args = [criterion,min_val,r[arg_index]]

    print(criterion)
    print(error_rates)
    print(standard_deviations)
    print(args)
    err.append(error_rates)
    std.append(standard_deviations)


for i, criterion in enumerate(criterions):
    plt.title(criterion)
    plt.plot(r,err[i])
    plt.errorbar(r, err[i], yerr=np.minimum(std[i]*2,err[i]), fmt='.r',lolims=True)
    plt.errorbar(r, err[i], yerr=np.minimum(std[i],err[i]), fmt='.k',lolims=True)

    plt.errorbar(r, err[i], yerr=np.minimum(std[i]*2,1-err[i]), fmt='.r',uplims=True)
    plt.errorbar(r, err[i], yerr=np.minimum(std[i],1-err[i]), fmt='.k',uplims=True)

    plt.show()
