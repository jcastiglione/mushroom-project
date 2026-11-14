import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

pois_train = pd.read_csv('data/mushroom_poisonous_train.csv')
pois_test = pd.read_csv('data/mushroom_poisonous_test.csv')
habi_train = pd.read_csv('data/mushroom_habitat_train.csv')
habi_test = pd.read_csv('data/mushroom_habitat_test.csv')

pois_r_train = pd.read_csv('data/mushroom_poisonous_reduced_train.csv')
pois_r_test = pd.read_csv('data/mushroom_poisonous_reduced_test.csv')
habi_r_train = pd.read_csv('data/mushroom_habitat_reduced_train.csv')
habi_r_test = pd.read_csv('data/mushroom_habitat_reduced_test.csv')


def prepare(X_train, X_other):
    # Impute + one-hot encode. Fit on X_train only.

    # One-hot (not ordinal) encoding is used since SVM kernels work on
    # distances/dot products, and ordinal codes would imply a false ordering
    # between categories. Categories unseen in training become all zeros.
 
    imputer = SimpleImputer(strategy='most_frequent')
    X_train = imputer.fit_transform(X_train)
    X_other = imputer.transform(X_other)

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train = encoder.fit_transform(X_train)
    X_other = encoder.transform(X_other)
    return X_train, X_other


def svm_model(df_train, df_test, target, kernel_values, filename,
              plot_filename, title):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        pass

    def log(message=""):
        print(message)
        with open(filename, 'a') as f:
            f.write(str(message) + '\n')

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    cv_start = time.time()

    for kernel in kernel_values:
        fold_accuracies = []

        for train_index, val_index in kf.split(X_train):
            X_fold_train = X_train.iloc[train_index]
            y_fold_train = y_train.iloc[train_index]
            X_fold_val = X_train.iloc[val_index]
            y_fold_val = y_train.iloc[val_index]

            X_fold_train, X_fold_val = prepare(X_fold_train, X_fold_val)

            model = SVC(kernel=kernel, C=1.0, gamma='scale', degree=3)
            model.fit(X_fold_train, y_fold_train)
            predictions = model.predict(X_fold_val)
            fold_accuracies.append(accuracy_score(y_fold_val, predictions))

        resultsrow = {'Hyperparameter (kernel)': kernel}
        for i in range(n_splits):
            resultsrow[f'Fold {i + 1} Accuracy'] = fold_accuracies[i]
        resultsrow['Average Accuracy'] = np.mean(fold_accuracies)
        results.append(resultsrow)

    cv_runtime = time.time() - cv_start
    results_df = pd.DataFrame(results)

    log(f"{n_splits}-FOLD CROSS-VALIDATION RESULTS")
    log(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    log(f"\nCross-validation runtime: {cv_runtime:.4f} seconds")

    best_kernel = results_df.loc[results_df['Average Accuracy'].idxmax(), 'Hyperparameter (kernel)']
    best_cv_accuracy = results_df['Average Accuracy'].max()

    log("\nBEST HYPERPARAMETER")
    log(f"Best kernel: {best_kernel}")
    log(f"Best CV accuracy: {best_cv_accuracy:.4f}")
    log(f"Best CV accuracy (%): {best_cv_accuracy * 100:.2f}%")

    # ---- CV visualization (bars, since kernel is categorical) ----
    fold_cols = [f'Fold {i + 1} Accuracy' for i in range(n_splits)]
    x_pos = np.arange(len(kernel_values))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x_pos, results_df['Average Accuracy'], color='tab:blue', alpha=0.7, label='Mean CV accuracy')
    for col in fold_cols:
        ax.scatter(x_pos, results_df[col], color='black', alpha=0.6, s=20, label='Individual folds' if col == fold_cols[0] else None)
    lo = results_df[fold_cols].min().min()
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(kernel_values)
    ax.set_xlabel('SVM kernel')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'SVM CV: {title}')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename, dpi=150)
    plt.close(fig)

    # Retrain on full training set (step 3)
    retrain_start = time.time()
    X_train_enc, X_test_enc = prepare(X_train, X_test)
    best_model = SVC(kernel=best_kernel, C=1.0, gamma='scale', degree=3)
    best_model.fit(X_train_enc, y_train)
    retrain_runtime = time.time() - retrain_start

    log("\nFINAL MODEL TRAINING")
    log(f"Selected kernel: {best_kernel}")
    log(f"Retraining runtime: {retrain_runtime:.4f} seconds")

    # Predict on held-out test set (step 4)
    predict_start = time.time()
    test_predictions = best_model.predict(X_test_enc)
    predict_runtime = time.time() - predict_start
    test_accuracy = accuracy_score(y_test, test_predictions)

    log("\nFINAL TEST SET RESULTS")
    log(f"Best hyperparameter (kernel): {best_kernel}")
    log(f"Test set accuracy: {test_accuracy:.4f}")
    log(f"Test set accuracy (%): {test_accuracy * 100:.2f}%")
    log(f"Prediction runtime: {predict_runtime:.4f} seconds")

    log("\nRUNTIME SUMMARY")
    log(f"Cross-validation runtime: {cv_runtime:.4f} seconds")
    log(f"Final model training runtime: {retrain_runtime:.4f} seconds")
    log(f"Final prediction runtime: {predict_runtime:.4f} seconds")
    log(f"Total runtime: {cv_runtime + retrain_runtime + predict_runtime:.4f} seconds")


KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']

svm_model(pois_train, pois_test, 'poisonous', KERNELS,
          filename="results/svm/poisonous.txt",
          plot_filename="results/svm/poisonous_cv.png",
          title="Poisonous (full features)")
svm_model(habi_train, habi_test, 'habitat', KERNELS,
          filename="results/svm/habitat.txt",
          plot_filename="results/svm/habitat_cv.png",
          title="Habitat (full features)")

svm_model(pois_r_train, pois_r_test, 'poisonous', KERNELS,
          filename="results/svm/poisonous_reduced.txt",
          plot_filename="results/svm/poisonous_reduced_cv.png",
          title="Poisonous (reduced features)")
svm_model(habi_r_train, habi_r_test, 'habitat', KERNELS,
          filename="results/svm/habitat_reduced.txt",
          plot_filename="results/svm/habitat_reduced_cv.png",
          title="Habitat (reduced features)")