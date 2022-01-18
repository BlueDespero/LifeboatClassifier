import itertools

import numpy as np
import pandas as pd


def get_data(path, encoder=None, cols=None, drop_invariant=True) -> (pd.DataFrame, pd.DataFrame):
    if not isinstance(path, list):
        path = [path]

    x_list = []
    y_list = []

    for p in path:
        data = pd.read_csv(p)
        xs = data[data.columns.difference(['Survived'])]
        ys = data['Survived'] if 'Survived' in data.columns else None

        x_list.append(xs)
        y_list.append(ys)

    if encoder is not None:
        x_merged = pd.concat(x_list)
        splits = np.cumsum([0] + [x.shape[0] for x in x_list])

        enc = encoder(cols=cols, drop_invariant=drop_invariant, return_df=True)
        enc.fit(x_merged)
        x_merged_transformed = enc.transform(x_merged)
        x_list = [x_merged_transformed.iloc[splits[i]:splits[i + 1]] for i in range(len(splits)-1)]

    return list(itertools.chain(*[[x, y] for x, y in zip(x_list, y_list)]))


def group(series, dict_with_labels={}):
    l = []
    counter = 0
    for el in series:
        if el not in dict_with_labels:
            dict_with_labels[el] = counter
            counter += 1
        l.append(dict_with_labels[el])
    return pd.Series(l, name=series.name), dict_with_labels


if __name__ == '__main__':
    # Training set

    # Importing dataframe
    df = pd.read_csv('../data/train.csv')

    # Covering Nans in Age to average
    df['Age'].fillna((df['Age'].mean()), inplace=True)

    # Converting Nans in cabins to new class '0'
    df['Cabin'].fillna((0), inplace=True)

    # As I see it, the only information that name can provide is that people in one cluster (based on surname)
    # may share same fate (or at least are somehow correlated). So I removed everything but surname.
    df['Name'] = df['Name'].apply(lambda full_name: full_name.split(',')[0])

    print("Training set:")
    print(df.head().to_string())

    # Changing each unique categorical value to unique number.
    df['Name'], d1 = group(df['Name'])
    df['Sex'], d2 = group(df['Sex'])
    df['Ticket'], d3 = group(df['Ticket'])
    df['Cabin'], d4 = group(df['Cabin'])
    df['Embarked'], d5 = group(df['Embarked'])

    print(df.head().to_string())

    # Saving new dataframe
    # df.to_csv('../data/train_num.csv', index=False)

    # -----------------------------------------------------------------------------------------------------------
    print("-" * 80 + "\nTest set:")

    # Test set and the same as previous transformation

    # Importing dataframe
    df = pd.read_csv('../data/test.csv')

    df['Age'].fillna((df['Age'].mean()), inplace=True)
    df['Fare'].fillna((df['Fare'].mean()), inplace=True)
    df['Cabin'].fillna((0), inplace=True)
    df['Name'] = df['Name'].apply(lambda full_name: full_name.split(',')[0])

    print(df.head().to_string())

    df['Name'], _ = group(df['Name'], d1)
    df['Sex'], _ = group(df['Sex'], d2)
    df['Ticket'], _ = group(df['Ticket'], d3)
    df['Cabin'], _ = group(df['Cabin'], d4)
    df['Embarked'], _ = group(df['Embarked'], d5)

    print(df.head().to_string())

    # Saving new dataframe
    # df.to_csv('../data/test_num.csv', index=False)
