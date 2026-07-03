import os
import sys
import time
import importlib.util
import inspect
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def discover_and_run_scenarios(base_dir: str):
    results = []
    base_path = Path(base_dir)
    
    for category in ["classification", "regression", "edge_cases", "production"]:
        cat_dir = base_path / category
        if not cat_dir.exists():
            continue
            
        for file in cat_dir.glob("test_*.py"):
            module_name = f"validation.{category}.{file.stem}"
            
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("run_validation_"):
                    console.print(f"[cyan]Running scenario:[/cyan] {module_name}.{name}")
                    start_time = time.time()
                    try:
                        report = obj()
                        status = "PASS"
                        error = None
                    except Exception as e:
                        status = "FAIL"
                        error = str(e)
                    duration = time.time() - start_time
                    
                    results.append({
                        "category": category,
                        "module": file.stem,
                        "function": name,
                        "status": status,
                        "error": error,
                        "duration_s": duration
                    })
                    
                    if status == "PASS":
                        console.print(f"[green]PASS[/green] ({duration:.2f}s)\n")
                    else:
                        console.print(f"[red]FAIL[/red]: {error}\n")
                        
    return results

def generate_reports(results, out_dir="validation/reports"):
    os.makedirs(out_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(out_dir, "validation_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Markdown & Rich Table
    md_path = os.path.join(out_dir, "validation_summary.md")
    
    table = Table(title="ModelDoctor Validation Benchmark Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Scenario", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Duration (s)", justify="right")
    
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    
    with open(md_path, "w") as f:
        f.write("# ModelDoctor Validation Benchmark Summary\n\n")
        f.write(f"**Total Scenarios:** {total}\n")
        f.write(f"**Passed:** {passed}\n")
        f.write(f"**Failed:** {total - passed}\n")
        f.write(f"**Pass Rate:** {(passed/total)*100:.1f}%\n\n")
        
        f.write("| Category | Scenario | Status | Duration (s) |\n")
        f.write("|----------|----------|--------|--------------|\n")
        
        for r in results:
            cat = r["category"]
            scen = r["function"].replace("run_validation_", "")
            stat = r["status"]
            dur = r["duration_s"]
            
            f.write(f"| {cat} | {scen} | {stat} | {dur:.2f} |\n")
            
            status_color = "green" if stat == "PASS" else "red"
            table.add_row(cat, scen, f"[{status_color}]{stat}[/{status_color}]", f"{dur:.2f}")

    console.print(table)
    console.print(f"\n[bold]Validation Summary saved to {out_dir}/[/bold]")

if __name__ == "__main__":
    console.print("[bold underline]ModelDoctor Benchmark Runner[/bold underline]\n")
    
    # Ensure working directory is project root
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)
    
    # Add root to python path to resolve 'validation.utils' correctly
    sys.path.insert(0, str(root_dir))
    
    results = discover_and_run_scenarios("validation")
    generate_reports(results)
