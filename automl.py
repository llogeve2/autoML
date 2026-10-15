import sklearn as sk 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class Automl():
    
    def __init__(self):
        # 1. Utilisation d'une LISTE [...] au lieu d'un SET {...} pour pouvoir l'indexer
        self.list_model = [
            sk.linear_model.LinearRegression(),
            sk.linear_model.SGDRegressor(),
            sk.linear_model.LogisticRegression(),
        ]
        self.type_methode = ["regression", "regression", "classification"]

        # 2. Données factices valides pour tester .eval()
        # On génère des données simples pour que .fit() fonctionne et évite la NotFittedError
        X_dummy = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_reg_dummy = np.array([3, 5, 7, 9])
        y_clf_dummy = np.array([0, 0, 1, 1])

        # Entraînement rapide des modèles pour les besoins du test de .eval()
        self.list_model[0].fit(X_dummy, y_reg_dummy)
        self.list_model[1].fit(X_dummy, y_reg_dummy)
        self.list_model[2].fit(X_dummy, y_clf_dummy)

        self.Xtrain = X_dummy
        self.Xtest = X_dummy
        self.Xvalidation = X_dummy

        self.Ytrain = y_reg_dummy
        self.Ytest = y_reg_dummy # (Attention pour la classification, dans un vrai cas y_test serait discret)
        self.Yvalidation = y_reg_dummy

        self.colone_garder = []

        self.best_estimator = None
        self.best_estimator_name = None
        self.rank = None

        if not self.list_model:
            raise ValueError("La liste des modèles est vide, veuillez la remplir.")

    def fit(self, path):
        pass

    def eval(self):
        score_max = -np.inf
        
        for i in range(len(self.list_model)):
            model = self.list_model[i]
            score = model.score(self.Xvalidation, self.Yvalidation)
            if score_max < score:
                score_max = score 
                self.best_estimator = model
                self.best_estimator_name = model.__class__.__name__
                self.rank = i

        # 3. Correction : on prédit sur Xtest et non Ytest
        y_pred = self.best_estimator.predict(self.Xtest)

        if self.type_methode[self.rank] == "classification":
            class_report = classification_report(self.Ytest, y_pred, zero_division=0)
            conf_matrix = str(confusion_matrix(self.Ytest, y_pred))
            report = class_report + "\n" + conf_matrix
        else: 
            MAE = f"MAE  : {mean_absolute_error(self.Ytest, y_pred):.2f}\n"
            RMSE = f"RMSE : {np.sqrt(mean_squared_error(self.Ytest, y_pred)):.2f}\n"
            rCarre = f"R²   : {r2_score(self.Ytest, y_pred):.2f}"
            report = MAE + RMSE + rCarre

        # 4. Correction : écriture de la variable 'report' globale (évite le NameError)
        with open("output.txt", "w") as f:
            f.write(report)

        return self.best_estimator

    def predict(self, path):
        pass

# --- Pour tester directement ---
if __name__ == "__main__":
    auto = Automl()
    best_model = auto.eval()
    print("Modèle sélectionné avec succès :", auto.best_estimator_name)
    print("Contenu du fichier output.txt généré.")