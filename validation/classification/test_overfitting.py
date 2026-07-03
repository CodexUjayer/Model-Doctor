import modeldoctor as md
from sklearn.model_selection import train_test_split
from validation.utils.datasets import get_healthy_classification_dataset
from validation.utils.models import train_overfitted_classifier
from validation.utils.assertions import assert_doctor_finding

def run_validation_overfitting():
    # 1. Dataset
    X, y = get_healthy_classification_dataset(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train overfitted model
    model = train_overfitted_classifier(X_train, y_train)
    
    # 3. Diagnose
    report = md.diagnose(model, X_train, y_train, X_test, y_test)
    
    # 4. Assert OverfittingDoctor found an issue
    assert_doctor_finding(report, doctor_name="OverfittingDoctor", min_severity="critical")
    
    return report
