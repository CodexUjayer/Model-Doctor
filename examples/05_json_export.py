"""
05 - JSON Export

Demonstrates exporting the diagnostic report to JSON for programmatic
ingestion, monitoring systems, or historical tracking.
"""

import os
import json
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

    output_path = "model_health_report.json"
    print(f"Saving JSON report to {output_path}...")

    # Use to_json() which returns a JSON string; write it to disk directly.
    # (report.save() delegates to the HTML renderer for all formats in this build.)
    json_str = report.to_json()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    if os.path.exists(output_path):
        print(f"Successfully created: {os.path.abspath(output_path)}")

        # Demonstrate loading and inspecting the exported JSON
        print("\nLoading and inspecting the exported JSON:")
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # health_score is a nested object in the JSON
        print(f"  Health Score: {data['health_score']['overall']:.1f}")
        print(f"  Number of Diagnoses: {len(data['diagnoses'])}")
        total_findings = sum(len(d.get("findings") or []) for d in data["diagnoses"])
        print(f"  Total Findings: {total_findings}")
    else:
        print("Error: Report was not saved.")


if __name__ == "__main__":
    main()
