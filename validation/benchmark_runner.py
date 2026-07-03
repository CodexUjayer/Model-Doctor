"""Benchmark Runner for ModelDoctor Validation Laboratory."""

import time
import sys
import traceback
from typing import Dict, Any, List

import modeldoctor as md
from validation.registry import ScenarioRegistry
from validation.metrics import MetricsEngine
from validation.benchmark_report import BenchmarkReporter
from validation.dashboard.generator import DashboardGenerator
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)

class BenchmarkRunner:
    """Executes validation scenarios and compares results."""

    def __init__(self):
        self.registry = ScenarioRegistry()
        self.results: List[Dict[str, Any]] = []

    def run(self, output_dir: str = "validation/reports"):
        """Run all discovered scenarios."""
        print("Starting ModelDoctor Validation Laboratory...")
        self.registry.discover()
        scenarios = self.registry.get_all()
        
        if not scenarios:
            print("No scenarios found!")
            return

        print(f"Discovered {len(scenarios)} validation scenarios.")
        
        for scenario_cls in scenarios:
            self._run_scenario(scenario_cls)

        print("\nComputing metrics...")
        metrics = MetricsEngine.compute(self.results)
        
        print("Generating reports...")
        BenchmarkReporter.generate(self.results, metrics, output_dir)
        DashboardGenerator.generate(self.results, metrics, f"{output_dir}/dashboard.html")
        
        print("\nValidation Complete!")
        
        # Return success boolean based on pass rate
        return metrics.get("failed", 0) == 0

    def _run_scenario(self, scenario_cls):
        scenario = scenario_cls()
        print(f"Running [{scenario.category}] {scenario.name}...")
        
        # Set random seed implicitly via numpy if used, or just let scenario handle it
        import numpy as np
        np.random.seed(scenario.random_seed)

        start_time = time.time()
        result_record = {
            "category": scenario.category,
            "scenario": scenario.name,
            "status": "FAIL",
            "duration_s": 0.0,
            "error": None,
            "expected_passed": None,
            "actual_passed": None,
            "report": None
        }

        try:
            expected = scenario.expected()
            result_record["expected_passed"] = expected.passed
            
            X_train, X_test, y_train, y_test = scenario.build_dataset()
            model = scenario.build_model(X_train, y_train)
            
            # Run ModelDoctor
            report = md.diagnose(model, X_train, y_train, X_test, y_test)
            result_record["report"] = report
            
            # Check conditions
            passed, err_msg = self._verify_report(report, expected, scenario.category)
            
            result_record["status"] = "PASS" if passed else "FAIL"
            result_record["error"] = err_msg
            
            # We determine actual passed by looking at the specific dimension score or overall passed
            # A bit tricky without knowing exactly which doctor, but we can look for CRITICAL/ERROR findings
            actual_passed = True
            for d in report.diagnoses:
                if d.dimension.lower() == scenario.category.lower() or scenario.category.lower() in d.dimension.lower():
                    actual_passed = d.passed
                    break
            else:
                # If dimension not directly matched, fall back to global
                actual_passed = report.health_score.overall > 80

            result_record["actual_passed"] = actual_passed
            
            if passed:
                print(f"  [PASS] ({time.time() - start_time:.2f}s)")
            else:
                print(f"  [FAIL] {err_msg}")

        except Exception as e:
            result_record["status"] = "ERROR"
            result_record["error"] = str(e)
            result_record["actual_passed"] = False
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            
        result_record["duration_s"] = time.time() - start_time
        self.results.append(result_record)

    def _verify_report(self, report, expected, category: str) -> tuple[bool, str]:
        """Verify the generated report against expected outputs."""
        all_findings = []
        for d in report.diagnoses:
            if category.lower() in d.dimension.lower() or d.dimension.lower() in category.lower():
                for f in d.findings:
                    all_findings.append(f)

        # Check expected findings strings — search title, explanation, AND evidence keys/names
        if expected.findings:
            for exp_str in expected.findings:
                found = False
                exp_lower = exp_str.lower()
                for f in all_findings:
                    # Check title
                    if exp_lower in f.title.lower():
                        found = True
                        break
                    # Check explanation
                    if exp_lower in getattr(f, 'explanation', '').lower():
                        found = True
                        break
                    # Check evidence dict keys (evidence names like "High Target Correlation")
                    if hasattr(f, 'evidence') and f.evidence:
                        if any(exp_lower in k.lower() for k in f.evidence.keys()):
                            found = True
                            break
                    # Check structured_evidence names
                    if hasattr(f, 'structured_evidence') and f.structured_evidence:
                        if any(exp_lower in e.name.lower() for e in f.structured_evidence):
                            found = True
                            break
                if not found:
                    return False, f"Expected finding containing '{exp_str}' not found."

        # Check severity (any expected severity should be present)
        if expected.severity:
            actual_severities = [f.severity.value.lower() for f in all_findings]
            if not any(exp_sev.lower() in actual_severities for exp_sev in expected.severity):
                return False, f"Expected one of severities {expected.severity} but got {actual_severities}."

        # Check passed (look at CRITICAL or ERROR findings in the dimension)
        if expected.passed is not None:
            has_critical = any(f.severity.value.lower() in ["critical", "error"] for f in all_findings)
            if expected.passed and has_critical:
                return False, "Expected model to pass, but CRITICAL/ERROR findings were present."
            if not expected.passed and not all_findings:
                return False, "Expected model to fail, but no findings were present in dimension."

        return True, ""


if __name__ == "__main__":
    runner = BenchmarkRunner()
    success = runner.run()
    sys.exit(0 if success else 1)
