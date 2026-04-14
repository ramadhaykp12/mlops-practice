import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def run_train():
    # Setup MLflow
    mlflow.set_experiment("Iris_Experiment")
    
    with mlflow.start_run():
        # Load Data
        df = pd.read_csv('data/iris.csv')
        X = df.drop('species', axis=1)
        y = df['species']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameters
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)
        
        # Model Training
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Evaluation
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # Logging Metrics & Model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Training selesai. Akurasi: {acc}")

if __name__ == "__main__":
    run_train()