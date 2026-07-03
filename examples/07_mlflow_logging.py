"""
07 - MLflow Logging

This example demonstrates how to integrate ModelDoctor with MLflow.
It trains a model, creates a local MLflow run, and logs the report 
artifacts directly to MLflow.
"""

import mlflow
from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from modeldoctor.integrations.mlflow import log_report
from utils import get_classification_data

def main():
    print("Generating data and training model...")
    X_train, X_test, y_train, y_test = get_classification_data(random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print("Running ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)
    
    print("Starting MLflow run and logging report...")
    
    # We use a local tracking URI for the example
    mlflow.set_tracking_uri("file:./mlruns")
    
    with mlflow.start_run() as run:
        # Log a standard metric
        mlflow.log_metric("accuracy", report.model_passport.training_samples)
        
        # Log the ModelDoctor report (JSON, HTML, Markdown artifacts)
        log_report(report, artifact_path="modeldoctor")
        
        print(f"\nSuccessfully logged to MLflow Run ID: {run.info.run_id}")
        print("Artifacts logged:")
        print(" - modeldoctor/dashboard.html")
        print(" - modeldoctor/report.json")
        print(" - modeldoctor/summary.md")
        
    print("\nTo view the MLflow UI, run: mlflow ui")

if __name__ == "__main__":
    main()
