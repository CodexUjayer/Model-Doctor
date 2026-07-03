"""Benchmark Reporter."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

class BenchmarkReporter:
    """Generates Markdown, JSON, CSV, and HTML reports from validation results."""

    @staticmethod
    def generate(results: List[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        base_path = Path(output_dir)

        BenchmarkReporter._write_json(results, metrics, base_path / "validation_summary.json")
        BenchmarkReporter._write_csv(results, base_path / "validation_summary.csv")
        BenchmarkReporter._write_markdown(results, metrics, base_path / "validation_summary.md")
        # HTML dashboard is handled separately in dashboard module, but we can write a simple HTML here as fallback
        BenchmarkReporter._write_html(results, metrics, base_path / "validation_summary.html")

    @staticmethod
    def _write_json(results: List[Dict[str, Any]], metrics: Dict[str, Any], path: Path):
        clean_results = []
        for r in results:
            clean_r = r.copy()
            if "report" in clean_r:
                del clean_r["report"]
            clean_results.append(clean_r)
            
        data = {
            "metrics": metrics,
            "results": clean_results
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _write_csv(results: List[Dict[str, Any]], path: Path):
        import csv
        if not results:
            return
            
        keys = ["category", "scenario", "status", "duration_s", "expected_passed", "actual_passed", "error"]
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    @staticmethod
    def _write_markdown(results: List[Dict[str, Any]], metrics: Dict[str, Any], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("========================================\n\n")
            f.write("ModelDoctor Validation Laboratory\n\n")
            f.write("========================================\n\n")
            f.write(f"Total Scenarios: {metrics.get('total_scenarios', 0)}\n\n")
            f.write(f"Passed: {metrics.get('passed', 0)}\n\n")
            f.write(f"Failed: {metrics.get('failed', 0)}\n\n")
            f.write("----------------------------------------\n\n")
            f.write("Doctor Accuracy\n\n")
            for doctor, acc in metrics.get("per_doctor_accuracy", {}).items():
                f.write(f"{doctor.capitalize()}: {acc*100:.1f}%\n\n")
            f.write("----------------------------------------\n\n")
            f.write(f"False Positives: {metrics.get('false_positive_rate', 0)*100:.1f}%\n\n")
            f.write(f"False Negatives: {metrics.get('false_negative_rate', 0)*100:.1f}%\n\n")
            f.write(f"Average Confidence: {metrics.get('average_confidence', 0):.1f}%\n\n")
            f.write(f"Average Runtime: {metrics.get('average_runtime_s', 0):.2f} s\n\n")
            f.write(f"Coverage: {metrics.get('coverage', 0)} scenarios\n\n")
            
            ready = "READY" if metrics.get('overall_accuracy', 0) == 1.0 else "NOT READY"
            f.write(f"Release Readiness: {ready}\n")

    @staticmethod
    def _write_html(results: List[Dict[str, Any]], metrics: Dict[str, Any], path: Path):
        # A simple fallback HTML report matching the Markdown
        html = f"""
        <html>
        <head><title>ModelDoctor Validation Laboratory</title></head>
        <body style="font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>ModelDoctor Validation Laboratory</h1>
            <h2>Summary</h2>
            <p>Total Scenarios: {metrics.get('total_scenarios', 0)}</p>
            <p>Passed: {metrics.get('passed', 0)}</p>
            <p>Failed: {metrics.get('failed', 0)}</p>
            <h2>Doctor Accuracy</h2>
            <ul>
        """
        for doctor, acc in metrics.get("per_doctor_accuracy", {}).items():
            html += f"<li>{doctor.capitalize()}: {acc*100:.1f}%</li>"
            
        html += f"""
            </ul>
            <h2>Metrics</h2>
            <p>False Positives: {metrics.get('false_positive_rate', 0)*100:.1f}%</p>
            <p>False Negatives: {metrics.get('false_negative_rate', 0)*100:.1f}%</p>
            <p>Average Runtime: {metrics.get('average_runtime_s', 0):.2f} s</p>
        </body>
        </html>
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
