import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import time

pois_train = pd.read_csv('data/mushroom_poisonous_train.csv')
pois_test = pd.read_csv('data/mushroom_poisonous_test.csv')
habi_train = pd.read_csv('data/mushroom_habitat_train.csv')
habi_test = pd.read_csv('data/mushroom_habitat_test.csv')

pois_r_train = pd.read_csv('data/mushroom_poisonous_reduced_train.csv')
pois_r_test = pd.read_csv('data/mushroom_poisonous_reduced_test.csv')
habi_r_train = pd.read_csv('data/mushroom_habitat_reduced_train.csv')
habi_r_test = pd.read_csv('data/mushroom_habitat_reduced_test.csv')

def plot_cv(results_df, hp_col, plot_filename, title):
    fold_cols = [c for c in results_df.columns
                 if c.startswith('Fold') and c.endswith('Accuracy')]
    x_pos = np.arange(len(results_df))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in fold_cols:
        ax.scatter(x_pos, results_df[col], color='gray', alpha=0.5, s=25)
    ax.plot(x_pos, results_df['Average Accuracy'], marker='o', linewidth=2,
            color='tab:blue', label='Mean CV accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df[hp_col].astype(str))
    ax.set_xlabel(hp_col)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=150)
    plt.close(fig)

def randforest_model(df_train, df_test, depth_values, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    cv_start = time.time()

    for depth in depth_values:

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

            model = RandomForestClassifier(
                max_depth=depth,
                random_state=42
            )

            model.fit(X_fold_train, y_fold_train)
            predictions = model.predict(X_fold_val)
            accuracy = accuracy_score(y_fold_val, predictions)
            fold_accuracies.append(accuracy)

        mean_accuracy = np.mean(fold_accuracies)

        resultsrow = {'Hyperparameter (max_depth)': depth}

        for i in range(len(fold_accuracies)):
            name = 'Fold ' + str(i+1) + ' Accuracy'
            resultsrow[name] = fold_accuracies[i]

        resultsrow['Average Accuracy'] = mean_accuracy

        results.append(resultsrow)

    cv_end = time.time()
    cv_runtime = cv_end - cv_start

    results_df = pd.DataFrame(results)

    results_df['Hyperparameter (max_depth)'] = results_df[
        'Hyperparameter (max_depth)'
    ].apply(lambda x: 'None' if pd.isna(x) else str(int(x)))
    plot_cv(results_df, 'Hyperparameter (max_depth)',
        filename.replace('.txt', '_cv.png'), 'Random Forest CV')

    log(f"5-FOLD CROSS-VALIDATION RESULTS")
    log(results_df.to_string(
        index=False,
        float_format=lambda x: f'{x:.4f}'
    ))

    log(f"\nCross-validation runtime: {cv_runtime:.4f} seconds")

    best_depth = depth_values[
        results_df['Average Accuracy'].idxmax()
    ]

    best_cv_accuracy = results_df['Average Accuracy'].max()

    log("\nBEST HYPERPARAMETER")
    log(f"Best max_depth: {best_depth}")
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

    best_model = RandomForestClassifier(
        max_depth=best_depth,
        random_state=42
    )

    best_model.fit(X_train_encoded, y_train)

    retrain_end = time.time()
    retrain_runtime = retrain_end - retrain_start

    log("\nFINAL MODEL TRAINING")
    log(f"Selected max_depth: {best_depth}")
    log(f"Retraining runtime: {retrain_runtime:.4f} seconds")

    predict_start = time.time()
    test_predictions = best_model.predict(X_test_encoded)
    predict_runtime = time.time() - predict_start
    test_accuracy = accuracy_score(y_test, test_predictions)

    log("\nFINAL TEST SET RESULTS")
    log(f"Best hyperparameter (max_depth): {best_depth}")
    log(f"Test set accuracy: {test_accuracy:.4f}")
    log(f"Test set accuracy (%): {test_accuracy * 100:.2f}%")
    log(f"Prediction runtime: {predict_runtime:.4f} seconds")

    log("\nRUNTIME SUMMARY")
    log(f"Cross-validation runtime: {cv_runtime:.4f} seconds")
    log(f"Final model training runtime: {retrain_runtime:.4f} seconds")
    log(f"Final prediction runtime: {predict_runtime:.4f} seconds")
    log(f"Total runtime: {cv_runtime + retrain_runtime + predict_runtime:.4f} seconds")

randforest_model(pois_train, pois_test,
                   depth_values=[3, 5, 12, 20, None],
                   filename="results/rf/poisonous.txt")
randforest_model(habi_train, habi_test,
                   depth_values=[3, 5, 12, 20, None],
                   filename="results/rf/habitat.txt")

randforest_model(pois_r_train, pois_r_test,
                   depth_values=[3, 5, 12, 20, None],
                   filename="results/rf/poisonous_reduced.txt")
randforest_model(habi_r_train, habi_r_test,
                   depth_values=[3, 5, 12, 20, None],
                   filename="results/rf/habitat_reduced.txt")