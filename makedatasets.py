import pandas as pd

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

pois = split_train_test("data/mushroom_poisonous", save = True)
habi = split_train_test('data/mushroom_habitat', save = True)

#Dimension Reduction
#Reduce dimensions by combining similar category classifiers into one larger classifier.
#Now instead of trying to match four different gill parts independently, the entire gill must be identical.
prefixes = ["cap", "gill", "stalk", "veil", "ring"]

pois_train_r = pois["train"]
pois_test_r = pois["test"]
habi_train_r = habi["train"]
habi_test_r = habi["test"]

for prefix in prefixes:
    cols = pois_train_r.filter(regex=f"^{prefix}").columns
    pois_train_r[prefix] = pois_train_r[cols].agg("".join, axis=1)

    cols = pois_test_r.filter(regex=f"^{prefix}").columns
    pois_test_r[prefix] = pois_test_r[cols].agg("".join, axis=1)

    cols = habi_train_r.filter(regex=f"^{prefix}").columns
    habi_train_r[prefix] = habi_train_r[cols].agg("".join, axis=1)

    cols = habi_test_r.filter(regex=f"^{prefix}").columns
    habi_test_r[prefix] = habi_test_r[cols].agg("".join, axis=1)

pois_drop = ["cap-shape", "cap-surface", "cap-color",
             "gill-attachment", "gill-spacing", "gill-size", "gill-color",
             "stalk-shape", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
             "veil-type", "veil-color",
             "ring-number", "ring-type"]
pois_train_r = pois_train_r.drop(columns=pois_drop)
pois_test_r = pois_test_r.drop(columns=pois_drop)

habi_drop = ["cap-shape", "cap-surface",
             "gill-attachment", "gill-spacing", "gill-size",
             "stalk-shape", "stalk-surface-above-ring", "stalk-surface-below-ring",
             "veil-type",
             "ring-number", "ring-type"]
habi_train_r = habi_train_r.drop(columns=habi_drop)
habi_test_r = habi_test_r.drop(columns=habi_drop)

pois_train_r.to_csv('data/mushroom_poisonous_reduced_train.csv', index=False)
pois_test_r.to_csv('data/mushroom_poisonous_reduced_test.csv', index=False)
habi_train_r.to_csv('data/mushroom_habitat_reduced_train.csv', index=False)
habi_test_r.to_csv('data/mushroom_habitat_reduced_test.csv', index=False)