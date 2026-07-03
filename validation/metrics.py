"""Metrics Engine for Validation Laboratory."""

from typing import Dict, List, Any
import numpy as np

class MetricsEngine:
    """Computes aggregate metrics from a list of benchmark results."""

    @staticmethod
    def compute(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASS")
        
        # True Positives: Scenario had expected findings and md.diagnose found them
        # False Positives: Scenario expected no findings but md.diagnose found some
        # False Negatives: Scenario expected findings but md.diagnose didn't find them
        # True Negatives: Scenario expected healthy and md.diagnose returned healthy
        
        tp = fn = fp = tn = 0
        
        per_doctor_total = {}
        per_doctor_passed = {}
        
        total_confidence = 0
        confidence_count = 0
        total_runtime = 0

        for r in results:
            cat = r["category"]
            per_doctor_total[cat] = per_doctor_total.get(cat, 0) + 1
            if r["status"] == "PASS":
                per_doctor_passed[cat] = per_doctor_passed.get(cat, 0) + 1
                
            total_runtime += r["duration_s"]

            expected = r["expected_passed"]
            actual = r["actual_passed"]

            if expected is False and actual is False:
                tp += 1
            elif expected is False and actual is True:
                fn += 1
            elif expected is True and actual is False:
                fp += 1
            elif expected is True and actual is True:
                tn += 1
                
            # Compute average confidence if available in findings
            report = r.get("report")
            if report and hasattr(report, "diagnoses"):
                for d in report.diagnoses:
                    for f in d.findings:
                        if hasattr(f, "confidence"):
                            conf_val = getattr(f.confidence, "value", str(f.confidence)).upper()
                            if conf_val == "HIGH":
                                total_confidence += 90
                            elif conf_val == "MEDIUM":
                                total_confidence += 60
                            elif conf_val == "LOW":
                                total_confidence += 30
                            else:
                                total_confidence += 50
                            confidence_count += 1

        accuracy = passed / total if total > 0 else 0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        avg_runtime = total_runtime / total if total > 0 else 0

        per_doctor_accuracy = {
            cat: (per_doctor_passed.get(cat, 0) / count) 
            for cat, count in per_doctor_total.items()
        }

        return {
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "overall_accuracy": accuracy,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "precision": precision,
            "recall": recall,
            "average_confidence": avg_confidence,
            "average_runtime_s": avg_runtime,
            "per_doctor_accuracy": per_doctor_accuracy,
            "coverage": total
        }
