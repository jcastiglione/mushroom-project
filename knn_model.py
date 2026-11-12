import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import time

pois_train = pd.read_csv('data/mushroom_poisonous_train.csv')
pois_test = pd.read_csv('data/mushroom_poisonous_test.csv')
habi_train = pd.read_csv('data/mushroom_habitat_train.csv')
habi_test = pd.read_csv('data/mushroom_habitat_test.csv')

def knn_model(df_train, df_test, k_values, filename):
    with open(filename, 'w') as f:
        pass

    def log(message=""):
        print(message)
        with open(filename, 'a') as f:
            f.write(str(message) + '\n')

    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    kf = KFold(n_splits=len(k_values), shuffle=True, random_state=42)
    results = []

    cv_start = time.time()

    for k in k_values:

        fold_accuracies = []

        for train_index, val_index in kf.split(df_train):
            training_fold = df_train.iloc[train_index]
            validation_fold = df_train.iloc[val_index]

            X_fold_train = training_fold.iloc[:, :-1]
            y_fold_train = training_fold.iloc[:, -1]

            X_fold_val = validation_fold.iloc[:, :-1]
            y_fold_val = validation_fold.iloc[:, -1]

            imputer = SimpleImputer(strategy='most_frequent')

            X_fold_train = imputer.fit_transform(X_fold_train)
            X_fold_val = imputer.transform(X_fold_val)

            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )

            X_fold_train = encoder.fit_transform(X_fold_train)
            X_fold_val = encoder.transform(X_fold_val)

            model = KNeighborsClassifier(
                n_neighbors=k,
                metric='hamming'
            )

            model.fit(X_fold_train, y_fold_train)
            predictions = model.predict(X_fold_val)
            accuracy = accuracy_score(y_fold_val, predictions)
            fold_accuracies.append(accuracy)

        mean_accuracy = np.mean(fold_accuracies)

        resultsrow = {'Hyperparameter (k)': k}
        for i in range(len(k_values)):
            name = 'Fold ' + str(i+1) + ' Accuracy'
            resultsrow[name] = fold_accuracies[i]
        resultsrow['Average Accuracy'] = mean_accuracy

        results.append(resultsrow)

    cv_end = time.time()
    cv_runtime = cv_end - cv_start

    results_df = pd.DataFrame(results)

    log(f"{len(k_values)}-FOLD CROSS-VALIDATION RESULTS")
    log(results_df.to_string(
        index=False,
        float_format=lambda x: f'{x:.4f}'
    ))

    log(f"\nCross-validation runtime: {cv_runtime:.4f} seconds")

    best_k = int(
        results_df.loc[
            results_df['Average Accuracy'].idxmax(),
            'Hyperparameter (k)']
    )
    best_cv_accuracy = results_df['Average Accuracy'].max()

    log("\nBEST HYPERPARAMETER")
    log(f"Best k: {best_k}")
    log(f"Best CV accuracy: {best_cv_accuracy:.4f}")
    log(f"Best CV accuracy (%): {best_cv_accuracy * 100:.2f}%")

    retrain_start = time.time()

    final_imputer = SimpleImputer(strategy='most_frequent')

    X_train_imputed = final_imputer.fit_transform(X_train)
    X_test_imputed = final_imputer.transform(X_test)

    final_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    X_train_encoded = final_encoder.fit_transform(X_train_imputed)
    X_test_encoded = final_encoder.transform(X_test_imputed)

    best_model = KNeighborsClassifier(
        n_neighbors=best_k,
        metric='hamming'
    )

    best_model.fit(X_train_encoded, y_train)

    retrain_end = time.time()
    retrain_runtime = retrain_end - retrain_start

    log("\nFINAL MODEL TRAINING")
    log(f"Selected k: {best_k}")
    log(f"Retraining runtime: {retrain_runtime:.4f} seconds")

    test_predictions = best_model.predict(X_test_encoded)
    test_accuracy = accuracy_score(y_test, test_predictions)

    log("\nFINAL TEST SET RESULTS")
    log(f"Best hyperparameter (k): {best_k}")
    log(f"Test set accuracy: {test_accuracy:.4f}")
    log(f"Test set accuracy (%): {test_accuracy * 100:.2f}%")

    log("\nRUNTIME SUMMARY")
    log(f"Cross-validation runtime: {cv_runtime:.4f} seconds")
    log(f"Final model training runtime: {retrain_runtime:.4f} seconds")
    log(f"Total runtime: {cv_runtime + retrain_runtime:.4f} seconds")

knn_model(pois_train, pois_test,
          k_values=[5, 10, 15, 20, 25],
          filename="results/knn/poisonous.txt")
knn_model(habi_train, habi_test,
          k_values=[1, 3, 5, 7, 10],
          filename="results/knn/habitats.txt")

#Dimension Reduction

#Reduce dimensions by combining similar category classifiers into one larger classifier.
#Now instead of trying to match four different gill parts independently, the entire gill must be identical.
prefixes = ["cap", "gill", "stalk", "veil", "ring"]

pois_train_r = pois_train.copy()
pois_test_r = pois_test.copy()
habi_train_r = habi_train.copy()
habi_test_r = habi_test.copy()

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

knn_model(pois_train_r, pois_test_r,
          k_values=[5, 10, 15, 20, 25],
          filename="results/knn/poisonous_reduced.txt")
knn_model(habi_train_r, habi_test_r,
          k_values=[1, 3, 5, 7, 10],
          filename="results/knn/habitats_reduced.txt")