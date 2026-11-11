import pandas as pd
import numpy as np

def split_train_test(csv_name, save = False):
    df = pd.read_csv(csv_name + '.csv')

    rows = df.shape[0]
    start = int(rows * 0.12)
    end = int(rows * 0.25)

    df_train = df.drop(df.index[start:end])
    df_test = df.iloc[start:end]

    if save:
        df_train.to_csv(csv_name + '_train.csv', index=False)
        df_test.to_csv(csv_name + '_test.csv', index=False)

    return {
        'train': df_train,
        'test': df_test,
    }

split_train_test("data/mushroom_poisonous", save = True)
split_train_test('data/mushroom_habitat', save = True)

def k_folds(df, k):
    return [fold for fold in np.array_split(df, k)]