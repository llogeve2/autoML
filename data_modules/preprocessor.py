from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:

    SEPARATORS = [",", ";", "\t", "|", r"\s+"]

    def __init__(self, folder_path):
        # Le chemin fourni désigne un dossier, par exemple data_A/
        self.folder_path = Path(folder_path)

        if not self.folder_path.is_dir():
            raise NotADirectoryError(
                f"Le dossier n'existe pas : {self.folder_path}"
            )

        # Les fichiers portent le nom du dossier
        dataset_name = self.folder_path.name

        self.data_path = self.folder_path / f"{dataset_name}.data"
        self.type_path = self.folder_path / f"{dataset_name}.type"
        self.solution_path = self.folder_path / f"{dataset_name}.solution"

        self.X = None
        self.y = None
        self.feature_types = None
        self.data_separator = None
        self.solution_separator = None
        self.preprocessor = None

    def load_types(self):
        self.feature_types = pd.read_csv(
            self.type_path,
            header=None,
            names=["type"]
        )["type"].str.strip().str.lower()

        allowed_types = {"numerical", "categorical"}

        if self.feature_types.empty:
            raise ValueError("Le fichier .type est vide.")

        invalid_types = set(self.feature_types) - allowed_types

        if invalid_types:
            raise ValueError(
                f"Types inconnus dans le fichier .type : "
                f"{invalid_types}"
            )

    def detect_separator(self, file_path, expected_columns=None):
        """
        Détecte le séparateur à partir des 10 premières lignes.

        Si expected_columns est fourni, le nombre de colonnes
        attendu permet de sélectionner le séparateur.

        Sinon, on privilégie le découpage régulier donnant le
        plus grand nombre de colonnes, notamment pour .solution.
        """
        candidates = []

        for sep in self.SEPARATORS:
            try:
                sample = pd.read_csv(
                    file_path,
                    sep=sep,
                    header=None,
                    nrows=10,
                    engine="python",
                    dtype=str,
                    skip_blank_lines=True
                )

                if sample.empty:
                    continue

                n_columns = sample.shape[1]

                # Vérification du nombre de colonnes attendu
                if (
                    expected_columns is not None
                    and n_columns != expected_columns
                ):
                    continue

                # On conserve les séparateurs qui donnent
                # un nombre de colonnes cohérent sur l'échantillon.
                candidates.append((sep, n_columns))

            except (ValueError, pd.errors.ParserError):
                continue

        if not candidates:
            raise ValueError(
                f"Aucun séparateur valide détecté pour {file_path}."
            )

        # Pour .data, le nombre de colonnes est connu grâce à .type.
        if expected_columns is not None:
            if len(candidates) > 1:
                # Plusieurs séparateurs peuvent produire le même
                # nombre de colonnes. On privilégie le découpage
                # le plus fin seulement si nécessaire.
                candidates.sort(
                    key=lambda candidate: (
                        candidate[1],
                        self.SEPARATORS.index(candidate[0])
                    ),
                    reverse=True
                )

            return candidates[0][0]

        # Pour .solution, aucun nombre de colonnes n'est connu :
        # on privilégie le découpage régulier le plus fin.
        candidates.sort(
            key=lambda candidate: (
                candidate[1],
                self.SEPARATORS.index(candidate[0])
            ),
            reverse=True
        )

        return candidates[0][0]

    def load_data(self):
        self.load_types()

        # Détection du séparateur des features
        self.data_separator = self.detect_separator(
            self.data_path,
            expected_columns=len(self.feature_types)
        )

        self.X = pd.read_csv(
            self.data_path,
            sep=self.data_separator,
            header=None,
            engine="python",
            skip_blank_lines=True
        )

        if self.X.shape[1] != len(self.feature_types):
            raise ValueError(
                f"{self.X.shape[1]} features chargées, mais "
                f"{len(self.feature_types)} types déclarés."
            )

        # Détection indépendante du séparateur des targets
        self.solution_separator = self.detect_separator(
            self.solution_path
        )

        self.y = pd.read_csv(
            self.solution_path,
            sep=self.solution_separator,
            header=None,
            engine="python",
            skip_blank_lines=True
        )

        if len(self.X) != len(self.y):
            raise ValueError(
                f"Nombre de lignes différent : X contient "
                f"{len(self.X)} lignes et y en contient {len(self.y)}."
            )

        print(f"Séparateur de X : {self.data_separator!r}")
        print(f"Séparateur de y : {self.solution_separator!r}")
        print(f"Nombre de features : {self.X.shape[1]}")
        print(f"Nombre de targets : {self.y.shape[1]}")

        return self.X, self.y

    def build_preprocessor(self):
        if self.X is None or self.feature_types is None:
            raise ValueError(
                "Il faut appeler load_data() avant "
                "build_preprocessor()."
            )

        numerical_columns = [
            self.X.columns[i]
            for i, t in enumerate(self.feature_types)
            if t == "numerical"
        ]

        categorical_columns = [
            self.X.columns[i]
            for i, t in enumerate(self.feature_types)
            if t == "categorical"
        ]

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("numeric", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ])

        return self.preprocessor