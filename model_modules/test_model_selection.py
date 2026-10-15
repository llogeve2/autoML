"""
Test de ModelSelector.

Usage :
    python test_model_selection.py              # données générées par sklearn
    python test_model_selection.py <dossier>    # vrai jeu (ex. .../data_A), nécessite data_modules/
"""

import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_selection import SEED, ModelSelector


def test(task, X, y, preprocessor):
    print(f"\n===== {task} =====")

    # On garde 20 % des données pour vérifier le modèle choisi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    selector = ModelSelector(task, preprocessor)
    selector.fit(X_train, y_train)

    # Score du meilleur modèle sur les données de test
    score = selector.best_model.score(X_test, selector.prepare_y(y_test))
    print(f"Score sur le test : {score:.4f}")


def main():
    if len(sys.argv) > 1:
        # Vrai jeu de données, chargé avec le Preprocessor de la branche preparation_donnees
        sys.path.insert(0, "../data_modules")
        from preprocessor import Preprocessor

        prep = Preprocessor(sys.argv[1])
        X, y = prep.load_data()
        test(prep.detect_task_type(), X, y, prep.build_preprocessor())
        return

    # Données générées : une classification et une régression
    X, y = make_classification(n_samples=500, n_features=10, random_state=SEED)
    X, y = pd.DataFrame(X), pd.DataFrame(y)
    preprocessor = ColumnTransformer([("numeric", StandardScaler(), list(X.columns))])
    test("binary_classification", X, y, preprocessor)

    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=SEED)
    X, y = pd.DataFrame(X), pd.DataFrame(y)
    preprocessor = ColumnTransformer([("numeric", StandardScaler(), list(X.columns))])
    test("regression", X, y, preprocessor)


if __name__ == "__main__":
    main()
