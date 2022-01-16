import pandas as pd


def get_data(path, encoder=None, cols=None, drop_invariant=True) -> pd.DataFrame:
    data = pd.read_csv(path)

    if encoder is not None:
        data = encoder.transform(data, cols=cols, drop_invariant=drop_invariant, return_df=True)

    return data


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
