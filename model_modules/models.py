
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
    LinearRegression
)

from sklearn.svm import LinearSVC

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
    MultiOutputClassifier,
    MultiOutputRegressor
)


class Models:
    """
    Sélectionne les modèles adaptés au type de tâche.

    Chaque appel à get_models() crée de nouvelles instances
    de modèles avec des hyperparamètres prédéfinis.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    def _classification_models(self):
        return {
            "Logistic Regression": LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            )
        }


    # --------------------------------------------------
    # Classification multilabel
    # --------------------------------------------------

    def _multilabel_models(self):
        return {
            "OneVsRest Logistic Regression": OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=self.random_state
                ),
                n_jobs=1
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            )
        }


    # --------------------------------------------------
    # Classification multioutput
    # --------------------------------------------------

    def _multioutput_classification_models(self):
        return {
            "MultiOutput Logistic Regression": MultiOutputClassifier(
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=self.random_state
                ),
                n_jobs=1
            ),
            "MultiOutput Random Forest": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            ),
            "MultiOutput Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            )
        }


    # --------------------------------------------------
    # Régression
    # --------------------------------------------------

    def _regression_models(self):
        return {
            "Ridge Regression": Ridge(alpha=1.0),
            "SGD Regressor": SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=0.0001,
                max_iter=2000,
                tol=1e-3,
                random_state=self.random_state
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            ),
            "Linear Regression": LinearRegression()
        }


    # --------------------------------------------------
    # Régression multioutput
    # --------------------------------------------------

    def _multioutput_regression_models(self):
        return {
            "MultiOutput Ridge": MultiOutputRegressor(
                Ridge(alpha=1.0),
                n_jobs=1
            ),
            "MultiOutput Random Forest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            ),
            "MultiOutput Extra Trees": ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=1,
                max_depth=15,
                random_state=self.random_state
            )
        }

    # --------------------------------------------------
    # Sélection
    # --------------------------------------------------

    def get_models(self, task_type):
        """
        Renvoie une liste de couples (nom, modèle)
        adaptée au type de tâche.
        """

        model_mapping = {
            "binary_classification": self._classification_models,
            "multiclass_classification": self._classification_models,
            "multilabel_classification": self._multilabel_models,
            "regression": self._regression_models,
            "multioutput_regression": self._multioutput_regression_models,
            "multioutput_classification": self._multioutput_classification_models
        }

        if task_type not in model_mapping:
            raise ValueError(
                f"Type de tâche non supporté : {task_type}"
            )

        return model_mapping[task_type]()
