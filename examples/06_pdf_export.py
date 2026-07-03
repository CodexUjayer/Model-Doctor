"""
06 - PDF Export

This example demonstrates the planned PDF export functionality.
Note: PDF export is currently scheduled for the v1.1 release and will 
raise a NotImplementedError in v1.0.
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
    
    print("Attempting to export PDF report...")
    try:
        # to_pdf is planned for the v1.1 API
        report.to_pdf("report.pdf")
        print("Successfully exported to PDF!")
    except NotImplementedError as e:
        print(f"Expected Exception: {e}")
        print("PDF export is currently tracked on the roadmap for v1.1.")

if __name__ == "__main__":
    main()
