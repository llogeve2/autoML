

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor:

    def __init__(self, path):
        path = Path(path)

        self.data_path = path.with_suffix(".data")
        self.data_type_path = path.with_suffix(".type")
        self.solution_path = path.with_suffix(".solution")

        self.X = None
        self.y = None
        self.feature_types = None
        self.separator = None

    def load_types(self):
        """Charge les types, un par ligne."""

        self.feature_types = pd.read_csv(
            self.data_type_path,
            header=None,
            names=["type"]
        )["type"].str.strip().str.lower()

        if self.feature_types.empty:
            raise ValueError("Le fichier .type est vide.")

        allowed_types = {"numerical", "categorical"}
        unknown = set(self.feature_types) - allowed_types

        if unknown:
            raise ValueError(f"Types inconnus : {unknown}")

    def detect_separator(self):
        """Cherche le séparateur compatible avec .type."""

        expected_features = len(self.feature_types)

        # Les candidats à tester
        separators = [",", ";", "\t", "|", r"\s+"]

        matches = []

        for sep in separators:
            try:
                sample = pd.read_csv(
                    self.data_path,
                    sep=sep,
                    header=None,
                    nrows=10,
                    engine="python"
                )

                n_features = sample.shape[1]

                print(f"sep={sep!r} -> {n_features} colonnes")

                if n_features == expected_features:
                    matches.append(sep)

            except (ValueError, pd.errors.ParserError):
                continue

        if len(matches) == 0:
            raise ValueError(
                f"Aucun séparateur ne produit "
                f"{expected_features} colonnes."
            )

        if len(matches) > 1:
            raise ValueError(
                f"Détection ambiguë : {matches}. "
                "Il faut examiner le fichier."
            )

        self.separator = matches[0]
        return self.separator

    def load_data(self):
        """Charge les features, types et targets."""

        self.load_types()
        sep = self.detect_separator()

        self.X = pd.read_csv(
            self.data_path,
            sep=sep,
            header=None,
            engine="python"
        )

        if self.X.shape[1] != len(self.feature_types):
            raise ValueError(
                "Le nombre de colonnes ne correspond pas "
                "au nombre de types."
            )

        self.y = pd.read_csv(
            self.solution_path,
            header=None
        )

        if len(self.X) != len(self.y):
            raise ValueError(
                "Le nombre de lignes de X et y diffère."
            )

        return self.X, self.y

	def build_preprocessor(self):

		if self.feature_types is None:
			raise ValueError("Les types de features ne sont pas chargés.")

		if len(self.feature_types) != X.shape[1]:
			raise ValueError(
				f"{len(self.feature_types)} types détectés, "
				f"mais {X.shape[1]} features présentes."
			)

		types = self.feature_types

		numeric_cols = [
			X.columns[i]
			for i, t in enumerate(types)
			if t == "numerical"
		]

		categorical_cols = [
			X.columns[i]
			for i, t in enumerate(types)
			if t == "categorical"
		]

		unknown_types = set(types) - {"numerical", "categorical"}

		if unknown_types:
			raise ValueError(f"Types inconnus : {unknown_types}")

		numeric_pipeline = Pipeline([
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler())
		])

		categorical_pipeline = Pipeline([
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore"))
		])

		self.preprocessor = ColumnTransformer([
			("numeric", numeric_pipeline, numeric_cols),
			("categorical", categorical_pipeline, categorical_cols)
		], remainder="drop")

		return self.preprocessor

