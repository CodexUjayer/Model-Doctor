"""
04 - HTML Report

This example demonstrates how to export a standalone HTML report
to disk, which is useful for CI/CD pipelines or sharing with stakeholders.
"""

import os
from sklearn.ensemble import RandomForestClassifier
import modeldoctor as md
from utils import get_classification_data

def main():
    print("Generating data and training model...")
    X_train, X_test, y_train, y_test = get_classification_data(random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print("Running ModelDoctor diagnostics...")
    report = md.diagnose(model, X_train, y_train, X_test, y_test)
    
    output_path = "model_health_report.html"
    print(f"Saving HTML report to {output_path}...")
    
    # Save the report using the unified save method. Format is inferred from extension.
    report.save(output_path)
    
    if os.path.exists(output_path):
        print(f"Successfully created: {os.path.abspath(output_path)}")
    else:
        print("Error: Report was not saved.")

if __name__ == "__main__":
    main()
