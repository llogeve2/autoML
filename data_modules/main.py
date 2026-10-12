from preprocessor import Preprocessor


def main():
    preprocessor = Preprocessor(
        "/info/corpus/ChallengeMachineLearning/data_A"
    )

    X, y = preprocessor.load_data()
    
    print("\n=== Chargement réussi ===")
    print(f"Observations : {X.shape[0]}")
    print(f"Séparateur détecté Observations: {preprocessor.data_separator}")
    print(f"Features : {X.shape[1]}")
    print(f"Targets : {y.shape[1]}")
    print(f"Séparateur détecté Targets: {preprocessor.solution_separator}")

    print("\n=== Types des features ===")
    print(preprocessor.feature_types.value_counts())

    print("\n=== Aperçu des data ===")
    print(X.head())

    print("\n=== Aperçu des targets ===")
    print(y.head())

    print("\n=== Préprocesseur ===")
    pipeline = preprocessor.build_preprocessor()
    print(pipeline)

    X_transformed = pipeline.fit_transform(X)

    print("\n=== Transformation réussie ===")
    print(f"Dimensions de X transformé : {X_transformed.shape}")

    print("\n=== Prédiction du type de tâche sur le jeu de données ===")
    print(f"-> {preprocessor.detect_task_type()}")

if __name__ == "__main__":
    main()