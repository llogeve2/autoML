import numpy as np
import sklearn as sk
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class Automl():
    
    def __init__(self):
        self.list_model = [
            sk.linear_model.LinearRegression(),
            sk.linear_model.SGDRegressor(),
            sk.linear_model.LogisticRegression(),
        ]
        self.type_methode = ["regression", "regression", "classification"]

        X_dummy = np.array([
            [1.5, 2.0], [2.0, 1.8], [1.0, 1.5], [2.5, 2.2],  # Groupe 1 (Classe 0)
            [6.0, 6.5], [7.0, 8.0], [6.5, 7.2], [8.0, 7.5]   # Groupe 2 (Classe 1)
        ])
        
        y_clf_dummy = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        self.list_model[0].fit(X_dummy, y_clf_dummy)
        self.list_model[1].fit(X_dummy, y_clf_dummy)
        self.list_model[2].fit(X_dummy, y_clf_dummy)

        self.Xtrain = X_dummy
        self.Xtest = X_dummy
        self.Xvalidation = X_dummy

        self.Ytrain = y_clf_dummy
        self.Ytest = y_clf_dummy
        self.Yvalidation = y_clf_dummy

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

        y_pred = self.best_estimator.predict(self.Xtest)

        if self.type_methode[self.rank] == "classification":
            class_report = classification_report(self.Ytest, y_pred, zero_division=0)
            conf_matrix = str(confusion_matrix(self.Ytest, y_pred))
            report = "\n" + class_report + "\n" + conf_matrix
        else: 
            MAE = f"MAE  : {mean_absolute_error(self.Ytest, y_pred):.2f}\n"
            RMSE = f"RMSE : {np.sqrt(mean_squared_error(self.Ytest, y_pred)):.2f}\n"
            rCarre = f"R²   : {r2_score(self.Ytest, y_pred):.2f}"
            report = MAE + RMSE + rCarre

        with open("output.txt", "w") as f:
            f.write(report)

        return self.best_estimator

    def predict(self, path):
        pass


if __name__ == "__main__":
    auto = Automl()
    best_model = auto.eval()
    print("Modèle sélectionné avec succès :", auto.best_estimator_name)
    print("Contenu du fichier output.txt généré.")