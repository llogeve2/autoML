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


# Chargement
preprocessor = Preprocessor("/info/corpus/ChallengeMachineLearning/data_A")
X, y = preprocessor.load_data()

# Détection de tâche par le Preprocessor
task = preprocessor.detect_task_type()
print(f"\n=== Type de tâche détecté : {task} ===")
print(f"X : {X.shape}")
print(f"y : {y.shape}")

if task not in CLASSIFICATION_TASKS | REGRESSION_TASKS:
    raise ValueError(f"Type de tâche non pris en charge : {task}")

if task in REGRESSION_TASKS:
    y = y.apply(pd.to_numeric, errors="raise").astype(float)

# Une cible simple doit être une Series pour sklearn.
if task in {"binary_classification", "multiclass_classification"}:
    y_model = y.iloc[:, 0]
else:
    y_model = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y_model, test_size=0.2, random_state=42
)

preprocessing_pipeline = preprocessor.build_preprocessor()

if task in CLASSIFICATION_TASKS:
    estimator = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model_name = "RandomForestClassifier"
else:
    estimator = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model_name = "RandomForestRegressor"

model = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("model", estimator),
])

print(f"\n=== Entraînement : {model_name} ===")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)