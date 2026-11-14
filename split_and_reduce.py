import pandas as pd
from sklearn.model_selection import train_test_split

PREFIXES = ["cap", "gill", "stalk", "veil", "ring"]

POIS_DROP = ["cap-shape", "cap-surface", "cap-color",
             "gill-attachment", "gill-spacing", "gill-size", "gill-color",
             "stalk-shape", "stalk-surface-above-ring", "stalk-surface-below-ring",
             "stalk-color-above-ring", "stalk-color-below-ring",
             "veil-type", "veil-color",
             "ring-number", "ring-type"]

HABI_DROP = ["cap-shape", "cap-surface",
             "gill-attachment", "gill-spacing", "gill-size",
             "stalk-shape", "stalk-surface-above-ring", "stalk-surface-below-ring",
             "veil-type",
             "ring-number", "ring-type"]


def reduce_features(df, target, drop_cols):
    """Merge related columns into one, then put the response column last."""
    out = df.copy()
    for prefix in PREFIXES:
        cols = [c for c in df.columns if c.startswith(prefix) and c != target]
        out[prefix] = df[cols].agg("".join, axis=1)
    out = out.drop(columns=drop_cols)
    cols = [c for c in out.columns if c != target] + [target]  # target last
    return out[cols]


def split_and_reduce(csv_name, target, drop_cols):
    df = pd.read_csv(csv_name + ".csv")

    # One stratified 10% split, used for BOTH full and reduced versions
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.10, stratify=df[target], random_state=42)

    full_train, full_test = df.loc[train_idx], df.loc[test_idx]
    red = reduce_features(df, target, drop_cols)
    red_train, red_test = red.loc[train_idx], red.loc[test_idx]

    full_train.to_csv(csv_name + "_train.csv", index=False)
    full_test.to_csv(csv_name + "_test.csv", index=False)
    red_train.to_csv(csv_name + "_reduced_train.csv", index=False)
    red_test.to_csv(csv_name + "_reduced_test.csv", index=False)

    print(f"\n{target}: train={len(full_train)}, test={len(full_test)} "
          f"({len(full_test) / len(df):.1%})")
    print(f"features: full={full_train.shape[1] - 1}, reduced={red_train.shape[1] - 1}")
    print(pd.concat([full_train[target].value_counts(normalize=True),
                     full_test[target].value_counts(normalize=True)],
                    axis=1, keys=["train", "test"]).round(3))


split_and_reduce("data/mushroom_poisonous", "poisonous", POIS_DROP)
split_and_reduce("data/mushroom_habitat", "habitat", HABI_DROP)