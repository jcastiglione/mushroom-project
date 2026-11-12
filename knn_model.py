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


X_train = pois_train.iloc[:, :-1]
y_train = pois_train.iloc[:, -1]

X_test = pois_test.iloc[:, :-1]
y_test = pois_test.iloc[:, -1]


k_values = [5, 15, 25, 35, 45]
kf = KFold(n_splits=5, shuffle=False)
results = []

cv_start = time.time()

for k in k_values:

    fold_accuracies = []

    for train_index, val_index in kf.split(pois_train):

        training_fold = pois_train.iloc[train_index]
        validation_fold = pois_train.iloc[val_index]

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

    results.append({
        'Hyperparameter (k)': k,
        'Fold 1 Accuracy': fold_accuracies[0],
        'Fold 2 Accuracy': fold_accuracies[1],
        'Fold 3 Accuracy': fold_accuracies[2],
        'Fold 4 Accuracy': fold_accuracies[3],
        'Fold 5 Accuracy': fold_accuracies[4],
        'Average Accuracy': mean_accuracy
    })

cv_end = time.time()
cv_runtime = cv_end - cv_start

results_df = pd.DataFrame(results)

print("5-FOLD CROSS-VALIDATION RESULTS")
print(results_df.to_string(
    index=False,
    float_format=lambda x: f'{x:.4f}'
))

print(f"\nCross-validation runtime: {cv_runtime:.4f} seconds")

best_k = int(
    results_df.loc[
        results_df['Average Accuracy'].idxmax(),
        'Hyperparameter (k)']
)
best_cv_accuracy = results_df['Average Accuracy'].max()

print("\nBEST HYPERPARAMETER")
print(f"Best k: {best_k}")
print(f"Best CV accuracy: {best_cv_accuracy:.4f}")
print(f"Best CV accuracy (%): {best_cv_accuracy * 100:.2f}%")


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

print("\nFINAL MODEL TRAINING")
print(f"Selected k: {best_k}")
print(f"Retraining runtime: {retrain_runtime:.4f} seconds")

test_predictions = best_model.predict(X_test_encoded)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\nFINAL TEST SET RESULTS")
print(f"Best hyperparameter (k): {best_k}")
print(f"Test set accuracy: {test_accuracy:.4f}")
print(f"Test set accuracy (%): {test_accuracy * 100:.2f}%")

print("\nRUNTIME SUMMARY")
print(f"Cross-validation runtime: {cv_runtime:.4f} seconds")
print(f"Final model training runtime: {retrain_runtime:.4f} seconds")
print(f"Total runtime: {cv_runtime + retrain_runtime:.4f} seconds")