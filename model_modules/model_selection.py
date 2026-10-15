"""
Partie : apprendre différents modèles et choisir le plus adapté.

Principe :
1. On choisit une liste de modèles selon le type de tâche
   (classification ou régression).
2. Chaque modèle est évalué par validation croisée sur les données
   d'entraînement.
3. On garde le modèle qui a le meilleur score moyen, puis on le
   ré-entraîne sur toutes les données d'entraînement.

Pour l'instant on utilise uniquement des modèles de sklearn.linear_model.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline


# Graine aléatoire fixée pour que les résultats soient reproductibles
SEED = 42

# Nombre de coeurs autorisés (variable d'environnement imposée par le sujet).
# On ne met jamais n_jobs=-1.
N_JOBS = int(os.environ.get("AUTOML_N_JOBS", 1))


class ModelSelector:

    def __init__(self, task, preprocessor):
        # task : type de tâche donné par Preprocessor.detect_task_type()
        # preprocessor : ColumnTransformer donné par Preprocessor.build_preprocessor()
        self.task = task
        self.preprocessor = preprocessor

        # Une tâche est une classification si son nom contient "classification"
        self.is_classification = "classification" in task

        # Une tâche a plusieurs cibles si c'est du multilabel ou du multioutput
        self.multi_target = task.startswith("multilabel") or task.startswith("multioutput")

        self.models = self.get_models()
        self.scores = {}        # nom du modèle -> score moyen en validation croisée
        self.best_name = None   # nom du meilleur modèle
        self.best_model = None  # pipeline (prétraitement + meilleur modèle) entraîné

    def get_models(self):
        """Renvoie un dictionnaire {nom: modèle} adapté au type de tâche."""
        if self.is_classification:
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
            }
            # LogisticRegression ne sait prédire qu'une seule cible :
            # pour plusieurs cibles, on entraîne un modèle par colonne.
            if self.multi_target:
                models = {name: MultiOutputClassifier(m) for name, m in models.items()}
        else:
            models = {
                "LinearRegression": LinearRegression(),
                "SGDRegressor": SGDRegressor(max_iter=1000, random_state=SEED),
            }
            # SGDRegressor ne gère qu'une seule cible (LinearRegression en gère plusieurs)
            if self.multi_target:
                models["SGDRegressor"] = MultiOutputRegressor(models["SGDRegressor"])
        return models

    def prepare_y(self, y):
        """Met la cible y au bon format pour scikit-learn."""
        # Une seule cible : sklearn attend un vecteur 1-D et non un tableau à 1 colonne
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        # En régression, la cible doit être numérique
        if not self.is_classification:
            y = np.asarray(y, dtype=float)
        return y

    def fit(self, X, y):
        """Compare les modèles par validation croisée et garde le meilleur."""
        y = self.prepare_y(y)

        # Métrique de comparaison : accuracy en classification, R² en régression
        # (dans les deux cas : plus le score est grand, meilleur est le modèle)
        scoring = "accuracy" if self.is_classification else "r2"

        for name, model in self.models.items():
            # Le prétraitement est mis dans le pipeline : il est donc appris
            # uniquement sur la partie entraînement de chaque fold (pas de fuite).
            pipeline = Pipeline([
                ("preprocessing", self.preprocessor),
                ("model", model),
            ])

            # Validation croisée en 5 parties
            scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring, n_jobs=N_JOBS)
            self.scores[name] = scores.mean()
            print(f"{name:20s} {scoring} = {scores.mean():.4f} (+/- {scores.std():.4f})")

        # Le meilleur modèle est celui qui a le plus grand score moyen
        self.best_name = max(self.scores, key=self.scores.get)
        print(f"Meilleur modèle : {self.best_name}")

        # On ré-entraîne le meilleur modèle sur toutes les données d'entraînement
        self.best_model = Pipeline([
            ("preprocessing", self.preprocessor),
            ("model", self.models[self.best_name]),
        ])
        self.best_model.fit(X, y)
        return self

    def predict(self, X):
        """Prédit avec le meilleur modèle trouvé par fit()."""
        if self.best_model is None:
            raise ValueError("Il faut appeler fit() avant predict().")
        return self.best_model.predict(X)
