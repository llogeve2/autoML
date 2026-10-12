
from preprocessor import Preprocessor


def main():
    preprocessor = Preprocessor(
        "/info/corpus/ChallengeMachineLearning/data_A/"
    )

    X, y = preprocessor.load_data()

    print("\n=== Chargement réussi ===")
    print(f"Séparateur détecté : {preprocessor.separator!r}")
    print(f"Nombre d'observations : {X.shape[0]}")
    print(f"Nombre de features : {X.shape[1]}")
    print(f"Nombre de targets : {y.shape[1]}")

    print("\n=== Types des features ===")
    print(preprocessor.feature_types.value_counts())

    print("\n=== Aperçu de X ===")
    print(X.head())

    print("\n=== Aperçu de y ===")
    print(y.head())

    pipeline = preprocessor.build_preprocessor()

    print("\n=== Préprocesseur construit ===")
    print(pipeline)

    X_transformed = pipeline.fit_transform(X)

    print("\n=== Transformation réussie ===")
    print(f"Dimensions après transformation : {X_transformed.shape}")


if __name__ == "__main__":
    main()
