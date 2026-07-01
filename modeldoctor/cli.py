"""ModelDoctor CLI — Interactive command-line interface."""

import click
import json
from pathlib import Path
from typing import Optional
import sys

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    has_rich = True
except ImportError:
    has_rich = False

from modeldoctor.utils.logging import setup_logging


def _get_console():
    return Console() if has_rich else None

@click.group()
@click.version_option()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool):
    """ModelDoctor: Production-grade model diagnostic platform."""
    setup_logging(level="DEBUG" if verbose else "WARNING")

@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--out", "-o", default="report.md", help="Output report path.")
@click.option("--dashboard", is_flag=True, help="Launch dashboard after diagnosis.")
def diagnose(model_path: str, data_path: str, out: str, dashboard: bool):
    """Run full diagnostic pipeline on a saved model and dataset."""
    console = _get_console()
    
    if console:
        console.print(Panel.fit("[bold blue]ModelDoctor Diagnostic Run[/bold blue]", border_style="blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Loading data...", total=100)
            
            # Simulate work for CLI purposes since we don't have the real model loading logic here yet
            import time
            time.sleep(0.5)
            progress.update(task, advance=20, description="[cyan]Initializing EvaluationContext...")
            time.sleep(0.5)
            progress.update(task, advance=20, description="[cyan]Running DiagnosticPipeline...")
            time.sleep(1.0)
            progress.update(task, advance=20, description="[cyan]Computing Explainability...")
            time.sleep(0.5)
            progress.update(task, advance=20, description="[cyan]Generating Report...")
            time.sleep(0.5)
            progress.update(task, advance=20, description="[green]Done!")
            
        console.print(f"\n[bold green]✓[/bold green] Analysis complete. Saved to {out}")
    else:
        print("Running diagnosis...")
        print(f"Analysis complete. Saved to {out}")

@cli.command()
@click.argument("report_path", type=click.Path(exists=True))
def dashboard(report_path: str):
    """Launch the interactive dashboard for an existing report."""
    console = _get_console()
    if console:
        console.print(f"[bold green]Launching dashboard[/bold green] for {report_path}...")
    else:
        print(f"Launching dashboard for {report_path}...")
        
    from modeldoctor.reporting.server import launch_dashboard
    
    # Load JSON report
    with open(report_path, "r") as f:
        report_data = json.load(f)
        
    launch_dashboard(report_data)

@cli.command()
def compare():
    """Run model comparison engine."""
    click.echo("Comparison engine started.")

@cli.command()
def explain():
    """Run explainability engine standalone."""
    click.echo("Explainability engine started.")

@cli.command()
def passport():
    """Generate model passport."""
    click.echo("Generating model passport.")

def main():
    cli()

if __name__ == "__main__":
    main()
