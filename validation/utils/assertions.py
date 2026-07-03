from typing import List

def assert_health_score_range(report, min_score=None, max_score=None):
    score = report.health_score.overall
    if min_score is not None:
        assert score >= min_score, f"Expected health score >= {min_score}, got {score}"
    if max_score is not None:
        assert score <= max_score, f"Expected health score <= {max_score}, got {score}"

def assert_doctor_finding(report, doctor_name: str, min_severity: str = None):
    # Find diagnoses by doctor name
    diagnoses = [d for d in report.diagnoses if d.doctor_name == doctor_name]
    assert len(diagnoses) > 0, f"Doctor {doctor_name} did not run or produced no diagnosis."
    
    diagnosis = diagnoses[0]
    if min_severity:
        # Check if any finding meets the severity
        severities = [f.severity.value if hasattr(f.severity, 'value') else str(f.severity).lower() for f in diagnosis.findings]
        assert min_severity in severities, f"Expected at least one finding with severity '{min_severity}' from {doctor_name}. Got {severities}"

def assert_no_critical_findings(report):
    for d in report.diagnoses:
        severities = [f.severity.value if hasattr(f.severity, 'value') else str(f.severity).lower() for f in d.findings]
        assert 'critical' not in severities, f"Unexpected critical finding in {d.doctor_name}"
