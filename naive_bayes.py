# Imports
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
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# Setting up trainning and testing data
pois_train = pd.read_csv('data/mushroom_poisonous_train.csv')
pois_test = pd.read_csv('data/mushroom_poisonous_test.csv')
habi_train = pd.read_csv('data/mushroom_habitat_train.csv')
habi_test = pd.read_csv('data/mushroom_habitat_test.csv')

pois_r_train = pd.read_csv('data/mushroom_poisonous_reduced_train.csv')
pois_r_test = pd.read_csv('data/mushroom_poisonous_reduced_test.csv')
habi_r_train = pd.read_csv('data/mushroom_habitat_reduced_train.csv')
habi_r_test = pd.read_csv('data/mushroom_habitat_reduced_test.csv')

def prepare (X_train, X_other):
    # Preparing the Categorical NB. We need to first remove all negative codes.
    # All categories are shifted by +1 and let category 0 for unseen in training
    imputer = SimpleImputer(strategy='most_frequent')
    X_train = imputer.fit_transform(X_train)
    X_other = imputer.transform(X_other)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value= -1)\

    X_train = encoder.fit_transform(X_train) + 1
    X_other = encoder.transform(X_other) + 1

    # Make sure model has extra space for extra category
    min_cats = [len(c) + 1 for c in encoder.categories_]
    return X_train.astype(int), X_other.astype(int), min_cats

def naive_bayes_model(df_train, df_test, target, alpha_values, filename, plot_filename, title):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        pass

    def log(message=""):
        print(message)
        with open(filename, 'a') as f:
            f.write(str(message) + '\n')

    # Use target column by Name
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    cv_start = time.time()
    for alpha in alpha_values:
        fold_accuracies = []

        for train_index, val_index in kf.split(X_train):
            X_fold_train = X_train.iloc[train_index]
            y_fold_train = y_train.iloc[train_index]
            X_fold_val = X_train.iloc[val_index]
            y_fold_val = y_train.iloc[val_index]

            X_fold_train, X_fold_val, min_cats = prepare(X_fold_train,
                                                         X_fold_val)

            model = CategoricalNB(alpha=alpha, min_categories=min_cats)
            model.fit(X_fold_train, y_fold_train)
            predictions = model.predict(X_fold_val)
            fold_accuracies.append(accuracy_score(y_fold_val, predictions))

        resultsrow = {'Hyperparameter (alpha)': alpha}
        for i in range(n_splits):
            resultsrow[f'Fold {i + 1} Accuracy'] = fold_accuracies[i]
        resultsrow['Average Accuracy'] = np.mean(fold_accuracies)
        results.append(resultsrow)

    cv_runtime = time.time() - cv_start
    results_df = pd.DataFrame(results)

    log(f"{n_splits}-FOLD CROSS-VALIDATION RESULTS")
    log(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    log(f"\nCross-validation runtime: {cv_runtime:.4f} seconds")

    best_alpha = float(results_df.loc[results_df['Average Accuracy'].idxmax(),'Hyperparameter (alpha)'])
    best_cv_accuracy = results_df['Average Accuracy'].max()

    log("\nBEST HYPERPARAMETER")
    log(f"Best alpha: {best_alpha}")
    log(f"Best CV accuracy: {best_cv_accuracy:.4f}")
    log(f"Best CV accuracy (%): {best_cv_accuracy * 100:.2f}%")

    # CV visulization
    fold_cols = [f'Fold {i + 1} Accuracy' for i in range(n_splits)]
    x_pos = np.arange(len(alpha_values))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in fold_cols:
        ax.scatter(x_pos, results_df[col], color='gray', alpha=0.5, s=25, label='Individual folds' if col == fold_cols[0] else None)
    ax.plot(x_pos, results_df['Average Accuracy'], marker='o', linewidth=2, color='tab:blue', label='Mean CV accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(a) for a in alpha_values])
    ax.set_xlabel('alpha (smoothing)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Naive Bayes CV: {title}')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    fig.savefig(plot_filename, dpi=150)
    plt.close(fig)

    # Retrain on full training set (step 3)
    retrain_start = time.time()
    X_train_enc, X_test_enc, min_cats = prepare(X_train, X_test)
    best_model = CategoricalNB(alpha=best_alpha, min_categories=min_cats)
    best_model.fit(X_train_enc, y_train)
    retrain_runtime = time.time() - retrain_start

    log("\nFINAL MODEL TRAINING")
    log(f"Selected alpha: {best_alpha}")
    log(f"Retraining runtime: {retrain_runtime:.4f} seconds")

    # Predict on held-out test set (step 4)
    predict_start = time.time()
    test_predictions = best_model.predict(X_test_enc)
    predict_runtime = time.time() - predict_start
    test_accuracy = accuracy_score(y_test, test_predictions)

    log("\nFINAL TEST SET RESULTS")
    log(f"Best hyperparameter (alpha): {best_alpha}")
    log(f"Test set accuracy: {test_accuracy:.4f}")
    log(f"Test set accuracy (%): {test_accuracy * 100:.2f}%")
    log(f"Prediction runtime: {predict_runtime:.4f} seconds")

    log("\nRUNTIME SUMMARY")
    log(f"Cross-validation runtime: {cv_runtime:.4f} seconds")
    log(f"Final model training runtime: {retrain_runtime:.4f} seconds")
    log(f"Final prediction runtime: {predict_runtime:.4f} seconds")
    log(f"Total runtime: {cv_runtime + retrain_runtime + predict_runtime:.4f} seconds")


ALPHAS = [0.001, 0.01, 0.1, 1, 10]

naive_bayes_model(pois_train, pois_test, 'poisonous', ALPHAS,
                  filename="results/nb/poisonous.txt",
                  plot_filename="results/nb/poisonous_cv.png",
                  title="Poisonous (full features)")
naive_bayes_model(habi_train, habi_test, 'habitat', ALPHAS,
                  filename="results/nb/habitat.txt",
                  plot_filename="results/nb/habitat_cv.png",
                  title="Habitat (full features)")

naive_bayes_model(pois_r_train, pois_r_test, 'poisonous', ALPHAS,
                  filename="results/nb/poisonous_reduced.txt",
                  plot_filename="results/nb/poisonous_reduced_cv.png",
                  title="Poisonous (reduced features)")
naive_bayes_model(habi_r_train, habi_r_test, 'habitat', ALPHAS,
                  filename="results/nb/habitat_reduced.txt",
                  plot_filename="results/nb/habitat_reduced_cv.png",
                  title="Habitat (reduced features)")