import pandas as pd
import numpy as np


def get_probs(series: pd.Series, target: pd.Series) -> dict:
    d = {key: [] for key in series.unique()}
    for el, tar in zip(series, target):
        d[el].append(tar)
    return {key: np.mean(d[key]) for key in d.keys() if len(d[key]) >= 2}


def naive_bayes_classifier(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    prob_survived = train['Survived'].mean()
    prob_died = 1 - prob_survived
    probs = {}
    predictions = []

    for name in test.columns[1:]:
        probs[name] = get_probs(train[name], train['Survived'])

    for _, row in test.iterrows():
        naive_part_survived = [np.log(probs[row_name].get(row_value, 0.5) + 1e-100) for row_name, row_value in
                               zip(row.index[1:], row[1:])]
        naive_part_died = [np.log((1 - probs[row_name].get(row_value, 0.5)) + 1e-100) for row_name, row_value in
                           zip(row.index[1:], row[1:])]

        prob_of_observation_cond_survived = np.exp(np.sum(naive_part_survived))
        prob_of_observation_cond_died = np.exp(np.sum(naive_part_died))

        probability_of_observation = prob_survived * prob_of_observation_cond_survived + prob_died * prob_of_observation_cond_died

        assumption = (prob_of_observation_cond_survived * prob_survived) / probability_of_observation

        predictions.append(int(assumption > 0.5))

    return pd.DataFrame({'PassengerId':test.iloc[:, 0],'Survived':pd.Series(predictions)})


if __name__ == '__main__':
    train = pd.read_csv('../../data/train_num.csv')
    test = pd.read_csv('../../data/test_num.csv')

    prediction = naive_bayes_classifier(train, test)
    print(prediction)

    # Saving solution
    # prediction.to_csv('../Predictions/naive_bayes_classifier.csv',index=False)
