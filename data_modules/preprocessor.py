from pathlib import Path

import pandas as pd
import numpy as np
from time import perf_counter
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import csr_matrix

def load_sparse_data(path):
    rows, cols, values = [], [], []
    n_rows = 0
    max_col = -1

    with open(path, "r") as f:
        for i, line in enumerate(f):
            items = line.split()

            if not items:
                continue

            n_rows += 1
            row = n_rows - 1

            for item in items:
                col, value = item.split(":", 1)
                col = int(col)

                if col < 0:
                    raise ValueError(
                        f"Indice de colonne négatif : {col}"
                    )

                rows.append(row)
                cols.append(col)
                values.append(float(value))
                max_col = max(max_col, col)

    if n_rows == 0:
        raise ValueError("Le fichier sparse est vide.")

    min_col = min(cols)

    if min_col == 1:
        cols = [col - 1 for col in cols]
        max_col -= 1
    elif min_col != 0:
        raise ValueError(
            f"Les indices sparse doivent commencer à 0 ou à 1, "
            f"mais le minimum trouvé est {min_col}."
        )

    return csr_matrix(
        (values, (rows, cols)),
        shape=(n_rows, max_col + 1),
        dtype=np.float64
    )

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

        allowed_types = {"numerical", "categorical", "binary"}

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

    def decode_one_hot_target(self):
        """Convertit une cible one-hot en une seule colonne de classes."""
        if self.y.shape[1] <= 1:
            return

        values = self.y.apply(pd.to_numeric, errors="coerce")

        if values.isna().any().any():
            return

        if not values.isin([0, 1]).all().all():
            return

        if not values.sum(axis=1).eq(1).all():
            return

        self.y = pd.DataFrame({
            "target": values.to_numpy().argmax(axis=1)
        })

    def is_sparse_file(self, path, n_lines=5):
        with open(path, "r") as f:
            checked = 0

            for line in f:
                if not line.strip():
                    continue

                items = line.split()

                # Toutes les valeurs doivent être de la forme indice:valeur
                try:
                    for item in items:
                        col, value = item.split(":", 1)
                        int(col)
                        float(value)
                except ValueError:
                    return False

                checked += 1
                if checked >= n_lines:
                    break

        return checked > 0

    def load_data(self):
        start = perf_counter()

        self.load_types()
        print(f"Lecture des types : {perf_counter() - start:.2f} s")

        t = perf_counter()
        if self.is_sparse_file(self.data_path):
            self.data_separator = "sparse"
            print(f"Détection du séparateur X (sparce): {perf_counter() - t:.2f} s")
            t = perf_counter()
            self.X = load_sparse_data(self.data_path)
        else:
            self.data_separator = self.detect_separator(
                self.data_path,
                expected_columns=len(self.feature_types)
            )
            print(f"Détection du séparateur X : {perf_counter() - t:.2f} s")
            t = perf_counter()

            self.X = pd.read_csv(
                self.data_path,
                sep=self.data_separator,
                header=None,
                engine="python",
                skip_blank_lines=True
            )
        print(f"Lecture de X : {perf_counter() - t:.2f} s")

        t = perf_counter()
        self.solution_separator = self.detect_separator(
            self.solution_path
        )
        print(f"Détection du séparateur y : {perf_counter() - t:.2f} s")

        t = perf_counter()
        self.y = pd.read_csv(
            self.solution_path,
            sep=self.solution_separator,
            header=None,
            engine="python",
            skip_blank_lines=True
        )
        print(f"Lecture de y : {perf_counter() - t:.2f} s")

        if self.X.shape[0] != len(self.y):
            raise ValueError(
                f"X contient {len(self.X)} lignes, "
                f"mais y en contient {len(self.y)}."
            )

        print(f"Temps total de chargement : {perf_counter() - start:.2f} s")

        self.decode_one_hot_target()

        return self.X, self.y

    def build_preprocessor(self):
        if self.X is None or self.feature_types is None:
            raise ValueError(
                "Il faut appeler load_data() avant "
                "build_preprocessor()."
            )

        numerical_columns = [
            i
            for i, t in enumerate(self.feature_types)
            if t == "numerical"
        ]

        categorical_columns = [
            i
            for i, t in enumerate(self.feature_types)
            if t in ["categorical", "binary"]
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

    def detect_task_type(self, override=None):
        """
        Détermine le type de tâche à partir de la cible y.

        Types possibles :
        - binary_classification
        - multiclass_classification
        - multilabel_classification
        - regression
        - multioutput_regression
        - multioutput_classification

        override permet de forcer explicitement le type de tâche
        lorsque les données ne permettent pas de le déduire.
        """
        if self.y is None:
            raise ValueError(
                "Il faut appeler load_data() avant detect_task_type()."
            )

        allowed_tasks = {
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
            "regression",
            "multioutput_regression",
            "multioutput_classification"
        }

        if override is not None:
            if override not in allowed_tasks:
                raise ValueError(
                    f"Tâche inconnue : {override}. "
                    f"Valeurs possibles : {sorted(allowed_tasks)}"
                )
            return override

        y = self.y.copy()

        # Tentative de conversion de toutes les colonnes en numérique
        numeric_y = y.apply(pd.to_numeric, errors="coerce")
        all_numeric = numeric_y.notna().all().all()

        # Plusieurs colonnes binaires : classification multilabel
        if y.shape[1] > 1:
            if all_numeric:
                values = set(np.unique(numeric_y.to_numpy()))
                if values.issubset({0, 1}):
                    return "multilabel_classification"

            # Plusieurs sorties numériques : régression multivariée
            if all_numeric:
                return "multioutput_regression"

            # Plusieurs sorties non numériques ou mixtes
            return "multioutput_classification"

        # Une seule colonne cible
        target = y.iloc[:, 0]

        # Une cible non numérique est considérée comme catégorielle
        if not all_numeric:
            return "binary_classification" if target.nunique() == 2 \
                else "multiclass_classification"

        values = numeric_y.iloc[:, 0].dropna()
        n_samples = len(values)
        n_classes = values.nunique()

        if n_classes <= 1:
            # Cible constante : choix par défaut
            return "regression"

        # Cible binaire
        if n_classes == 2:
            return "binary_classification"

        # Indices statistiques
        integer_target = np.allclose(
            values.to_numpy(),
            np.round(values.to_numpy())
        )

        cardinality_ratio = n_classes / n_samples

        # Seuil adaptatif selon la taille du dataset
        max_classes = min(
            50,
            max(10, int(np.sqrt(n_samples)))
        )

        # Classification multiclasse probable :
        # cible entière, cardinalité limitée et valeurs répétées
        if (
            integer_target
            and n_classes <= max_classes
            and cardinality_ratio <= 0.05
        ):
            return "multiclass_classification"

        # Par défaut : régression
        return "regression"