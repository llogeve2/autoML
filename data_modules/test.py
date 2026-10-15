import numpy as np
import pandas as pd

from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from model_modules.model_selection import SEED, ModelSelector


CLASSIFICATION_TASKS = {
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multioutput_classification",
}
REGRESSION_TASKS = {"regression", "multioutput_regression"}


def evaluate_classification(y_test, y_pred, task):
    """Affiche un rapport pour une cible simple ou un rapport par target."""
    if task in {"binary_classification", "multiclass_classification"}:
        y_true = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test
        print(classification_report(
            y_true, np.asarray(y_pred).ravel(), zero_division=0
        ))
        return

    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    for i, column in enumerate(y_test.columns):
        print(f"\n--- Rapport pour {column} ---")
        print(classification_report(
            y_test.iloc[:, i], y_pred[:, i], zero_division=0
        ))


def evaluate_regression(y_test, y_pred):
    """Affiche MAE, RMSE et R² pour une ou plusieurs targets."""
    y_true = (
        y_test.to_numpy(dtype=float)
        if isinstance(y_test, pd.DataFrame)
        else np.asarray(y_test, dtype=float)
    )
    y_pred = np.asarray(y_pred, dtype=float)

    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²   : {r2_score(y_true, y_pred, multioutput='uniform_average'):.4f}")

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        print("\nMétriques par target :")
        for i in range(y_true.shape[1]):
            print(
                f"Target_{i}: "
                f"MAE={mean_absolute_error(y_true[:, i], y_pred[:, i]):.4f}, "
                f"RMSE={np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])):.4f}, "
                f"R²={r2_score(y_true[:, i], y_pred[:, i]):.4f}"
            )

def test(task, X, y, preprocessor):
    print(f"\n===== {task} =====")

    # On garde 20 % des données pour vérifier le modèle choisi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    selector = ModelSelector(task, preprocessor)
    selector.fit(X_train, y_train)

    # Score du meilleur modèle sur les données de test
    #score = selector.best_model.score(X_test, selector.prepare_y(y_test))
    #print(f"Score sur le test : {score:.4f}")
    
    if task in CLASSIFICATION_TASKS:
        evaluate_classification(y_test, selector.predict(X_test), task)
    else:
        evaluate_regression(y_test, selector.predict(X_test))


# Chargement
letters = ['A','B','C','D','E','F'] # AJOUTER LES LETTRES SI D'AUTRES DATASETS SONT DISPOS

for letter in letters:
    print(f"\n =========================== DATASET {letter} ===========================\n\n")

    preprocessor = Preprocessor("/info/corpus/ChallengeMachineLearning/data_" + letter)
    X, y = preprocessor.load_data()

    # Détection de tâche par le Preprocessor
    task = preprocessor.detect_task_type()
    print(f"\n=== Type de tâche détecté : {task} ===")
    print(f"X : {X.shape}")
    print(f"y : {y.shape}")

    if task not in CLASSIFICATION_TASKS | REGRESSION_TASKS:
        raise ValueError(f"Type de tâche non pris en charge : {task}")

    test(task, X, y, preprocessor.build_preprocessor())