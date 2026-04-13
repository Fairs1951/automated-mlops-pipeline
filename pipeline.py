import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

class MLOpsPipeline:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def run_training(self, data_path):
        df = pd.read_csv(data_path)
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        with mlflow.start_run():
            n_estimators = 100
            model = RandomForestClassifier(n_estimators=n_estimators)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Run completed. Accuracy: {acc}")

if __name__ == "__main__":
    # pipeline = MLOpsPipeline("churn_prediction")
    pass
