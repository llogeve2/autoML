"""
Partie : apprendre différents modèles et choisir le plus adapté.

Principe :
1. On récupère une liste de modèles adaptée au type de tâche
   (voir model_modules/models.py).
2. Chaque modèle est évalué par validation croisée sur les données
   d'entraînement.
3. On garde le modèle qui a le meilleur score moyen, puis on le
   ré-entraîne sur toutes les données d'entraînement.
4. (tune) On optimise les hyperparamètres du meilleur modèle avec
   RandomizedSearchCV.
"""

import os
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from model_modules.models import Models


# Graine aléatoire fixée pour que les résultats soient reproductibles
SEED = 42

# Ressources imposées par le sujet (variables d'environnement).
# On ne met jamais n_jobs=-1.
N_JOBS = int(os.environ.get("AUTOML_N_JOBS", 1))
N_TRIALS = int(os.environ.get("AUTOML_N_TRIALS", 20))      # nombre d'essais d'optimisation
TIME_BUDGET = float(os.environ.get("AUTOML_TIME_BUDGET", 600))  # secondes pour tout le fit

# Pendant l'optimisation, on travaille sur au plus TUNE_MAX_ROWS lignes
# et avec 3 folds au lieu de 5, pour que ce soit beaucoup plus rapide.
TUNE_MAX_ROWS = 5000
TUNE_CV = 3


# --------------------------------------------------
# Espaces de recherche des hyperparamètres
# --------------------------------------------------
# Le modèle est dans un Pipeline sous le nom "model" : un paramètre
# s'écrit donc "model__<param>". Pour un modèle emballé dans
# OneVsRestClassifier ou MultiOutputClassifier/Regressor, on ajoute
# "estimator__" : "model__estimator__<param>".
#
# loguniform : on tire l'ordre de grandeur (0.001, 0.1, 10...) plutôt
# qu'une valeur exacte, car c'est ce qui compte pour C et alpha.

LOGISTIC_SPACE = {"C": loguniform(1e-3, 1e2)}
RIDGE_SPACE = {"alpha": loguniform(1e-3, 1e3)}
SGD_SPACE = {"alpha": loguniform(1e-6, 1e-1)}
TREES_SPACE = {
    "n_estimators": randint(100, 400),
    "max_depth": [None, 10, 15, 25],
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
}


def with_prefix(space, prefix):
    """Ajoute le préfixe du pipeline devant chaque nom de paramètre."""
    return {prefix + name: values for name, values in space.items()}


# Nom du modèle (celui de models.py) -> espace de recherche.
# "Linear Regression" n'a pas d'hyperparamètre : elle n'est pas optimisée.
PARAM_SPACES = {
    "Logistic Regression": with_prefix(LOGISTIC_SPACE, "model__"),
    "OneVsRest Logistic Regression": with_prefix(LOGISTIC_SPACE, "model__estimator__"),
    "MultiOutput Logistic Regression": with_prefix(LOGISTIC_SPACE, "model__estimator__"),
    "Ridge Regression": with_prefix(RIDGE_SPACE, "model__"),
    "MultiOutput Ridge": with_prefix(RIDGE_SPACE, "model__estimator__"),
    "SGD Regressor": with_prefix(SGD_SPACE, "model__"),
    "Random Forest": with_prefix(TREES_SPACE, "model__"),
    "Extra Trees": with_prefix(TREES_SPACE, "model__"),
    "MultiOutput Random Forest": with_prefix(TREES_SPACE, "model__"),
    "MultiOutput Extra Trees": with_prefix(TREES_SPACE, "model__"),
}


class ModelSelector:

    def __init__(self, task, preprocessor):
        # task : type de tâche donné par Preprocessor.detect_task_type()
        # preprocessor : ColumnTransformer donné par Preprocessor.build_preprocessor()
        self.task = task
        self.preprocessor = preprocessor

        # Une tâche est une classification si son nom contient "classification"
        self.is_classification = "classification" in task

        # Métrique de comparaison : accuracy en classification, R² en régression
        # (dans les deux cas : plus le score est grand, meilleur est le modèle)
        self.scoring = "accuracy" if self.is_classification else "r2"

        self.models = self.get_models()
        self.scores = {}        # nom du modèle -> score moyen en validation croisée
        self.times = {}         # nom du modèle -> durée de sa validation croisée (s)
        self.best_name = None   # nom du meilleur modèle
        self.best_model = None  # pipeline (prétraitement + meilleur modèle) entraîné
        self.best_params = {}   # hyperparamètres trouvés par tune()
        self.start_time = None  # début du fit, pour respecter TIME_BUDGET

    def get_models(self):
        """Renvoie un dictionnaire {nom: modèle} adapté au type de tâche."""
        # Les modèles mettent n_jobs=1 : le parallélisme se fait sur les folds
        # de la validation croisée (voir fit).
        return Models(random_state=SEED).get_models(self.task)

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
        self.start_time = perf_counter()
        y = self.prepare_y(y)
        scoring = self.scoring

        for name, model in self.models.items():
            # Le prétraitement est mis dans le pipeline : il est donc appris
            # uniquement sur la partie entraînement de chaque fold (pas de fuite).
            pipeline = Pipeline([
                ("preprocessing", self.preprocessor),
                ("model", model),
            ])

            print("Entrainement sur le modele " + name)
            start = perf_counter()

            # Validation croisée en 5 parties
            scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring, n_jobs=N_JOBS)
            self.scores[name] = scores.mean()
            self.times[name] = perf_counter() - start
            print(f"{name:20s} {scoring} = {scores.mean():.4f} (+/- {scores.std():.4f})")
            print(f"Temps écoulé : {self.times[name]:.2f}")

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

    def tune(self, X, y):
        """
        Optimise les hyperparamètres du meilleur modèle trouvé par fit().

        On tire au hasard N_TRIALS combinaisons dans PARAM_SPACES
        (RandomizedSearchCV), sur un sous-échantillon et en 3 folds pour
        aller vite, puis on ré-entraîne sur toutes les données avec la
        meilleure combinaison.
        """
        if self.best_model is None:
            raise ValueError("Il faut appeler fit() avant tune().")

        space = PARAM_SPACES.get(self.best_name)
        if space is None:
            print(f"{self.best_name} : pas d'hyperparamètre à optimiser")
            return self

        y = self.prepare_y(y)

        # 1. Sous-échantillon : on cherche les paramètres sur moins de lignes.
        #    resample fonctionne avec des DataFrame et des matrices creuses.
        n_rows = X.shape[0]
        if n_rows > TUNE_MAX_ROWS:
            X_tune, y_tune = resample(X, y, n_samples=TUNE_MAX_ROWS,
                                      replace=False, random_state=SEED)
        else:
            X_tune, y_tune = X, y

        # 2. Combien d'essais peut-on faire dans le temps restant ?
        #    Estimation d'un essai : durée de la validation croisée en 5 folds
        #    du modèle pendant fit(), ramenée à 3 folds et au sous-échantillon.
        remaining = TIME_BUDGET - (perf_counter() - self.start_time)
        trial_time = (self.times[self.best_name] * TUNE_CV / 5
                      * X_tune.shape[0] / n_rows)
        # Il faut garder du temps pour le ré-entraînement final sur toutes les données
        remaining -= self.times[self.best_name] / 5
        n_iter = min(N_TRIALS, int(remaining / max(trial_time, 1e-3)))

        if n_iter < 1:
            print("Pas assez de temps restant : pas d'optimisation")
            return self
        print(f"Optimisation de {self.best_name} : {n_iter} essais "
              f"sur {X_tune.shape[0]} lignes (temps restant : {remaining:.0f} s)")

        # 3. Recherche aléatoire, reproductible grâce à random_state
        search = RandomizedSearchCV(
            self.best_model,
            space,
            n_iter=n_iter,
            cv=TUNE_CV,
            scoring=self.scoring,
            n_jobs=N_JOBS,
            random_state=SEED,
            refit=False,   # on ré-entraîne nous-mêmes sur toutes les données
        )
        search.fit(X_tune, y_tune)

        # 4. Les paramètres par défaut sont évalués dans les mêmes conditions
        #    (même sous-échantillon, mêmes folds) : on ne garde l'optimisation
        #    que si elle fait vraiment mieux.
        default_score = cross_val_score(
            self.best_model, X_tune, y_tune, cv=TUNE_CV,
            scoring=self.scoring, n_jobs=N_JOBS).mean()
        print(f"Score paramètres par défaut : {default_score:.4f}")
        print(f"Score meilleurs paramètres  : {search.best_score_:.4f} "
              f"{search.best_params_}")

        if search.best_score_ <= default_score:
            print("L'optimisation ne fait pas mieux : on garde les paramètres par défaut")
            return self

        # 5. Ré-entraînement sur toutes les données avec les meilleurs paramètres
        self.best_params = search.best_params_
        self.best_model = clone(self.best_model).set_params(**self.best_params)
        self.best_model.fit(X, y)
        return self

    def predict(self, X):
        """Prédit avec le meilleur modèle trouvé par fit()."""
        if self.best_model is None:
            raise ValueError("Il faut appeler fit() avant predict().")
        return self.best_model.predict(X)
