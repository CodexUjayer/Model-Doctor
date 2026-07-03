"""
03 - Dashboard

This example demonstrates how to generate and launch the interactive 
HTML dashboard to explore the diagnostic report visually.
"""

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
    
    print("Launching interactive dashboard...")
    print("The dashboard will open in your default web browser.")
    # Opens the interactive HTML dashboard
    report.dashboard()
    
if __name__ == "__main__":
    main()
