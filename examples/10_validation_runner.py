"""
10 - Validation Runner

This example demonstrates how to programmatically run the ModelDoctor
Validation Laboratory to execute deterministic diagnostic benchmarks.
"""

import os
from validation.benchmark_runner import BenchmarkRunner

def main():
    print("Initializing ModelDoctor Validation Laboratory...")
    print("This will execute 54 deterministic edge-case benchmarks.")
    
    runner = BenchmarkRunner()
    
    # We pass fail_fast=False to run all scenarios even if one fails
    print("\nRunning benchmarks...")
    success_rate = runner.run_all(fail_fast=False)
    
    print("\n--- Validation Complete ---")
    print(f"Overall Accuracy: {success_rate * 100:.1f}%")
    
    # Check if the runner generated reports
    if os.path.exists("validation/reports"):
        print("\nDetailed validation reports have been saved to: validation/reports/")
        print(" - validation/reports/validation_summary.html")
        print(" - validation/reports/validation_summary.md")
        print(" - validation/reports/validation_results.json")

if __name__ == "__main__":
    main()
