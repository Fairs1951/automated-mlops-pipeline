import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsPipeline:
    """
    End-to-end MLOps pipeline for model training, tracking, and deployment.
    """
    def __init__(self, experiment_name: str, model_name: str = "random_forest_model"):
        self.experiment_name = experiment_name
        self.model_name = model_name
        mlflow.set_experiment(experiment_name)

    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads and prepares data for training.
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Simple preprocessing
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_with_tracking(self, X_train, X_test, y_train, y_test, params: Dict[str, Any]):
        """
        Trains the model and tracks experiments using MLflow.
        """
        with mlflow.start_run() as run:
            logger.info("Starting model training...")
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predictions and Evaluation
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
            
            # Log parameters, metrics, and model
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, self.model_name)
            
            logger.info(f"Model training complete. Metrics: {metrics}")
            return run.info.run_id, model

    def hyperparameter_tuning(self, X_train, y_train, param_grid: Dict[str, List[Any]]):
        """
        Performs grid search for hyperparameter tuning.
        """
        logger.info("Starting hyperparameter tuning...")
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        return grid_search.best_params_

    def save_model_locally(self, model, path: str):
        """
        Saves the trained model to a local directory.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        joblib.dump(model, path)
        logger.info(f"Model saved locally at {path}")

    def load_model_locally(self, path: str):
        """
        Loads a model from a local directory.
        """
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model

if __name__ == "__main__":
    # Test pipeline with dummy data
    pipeline = MLOpsPipeline("customer_churn_prediction")
    data = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
    data['target'] = np.random.randint(0, 2, 100)
    data.to_csv("dummy_data.csv", index=False)
    
    X_train, X_test, y_train, y_test = pipeline.prepare_data("dummy_data.csv")
    params = {"n_estimators": 100, "max_depth": 10}
    run_id, model = pipeline.train_with_tracking(X_train, X_test, y_train, y_test, params)
    print(f"MLflow Run ID: {run_id}")
