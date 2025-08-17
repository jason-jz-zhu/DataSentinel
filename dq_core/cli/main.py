"""Main CLI entry point for DataSentinel."""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from dq_core.config import load_config
from dq_core.models import Dataset, Check, CheckSeverity, DQDimension
from dq_core.registry import adapter_registry
from dq_core.scoring.score import DQScorer
from dq_core.profiling.history_store import MetricsHistoryStore


console = Console()


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)]
    )


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """DataSentinel: Enterprise Data Quality Framework."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    setup_logging(debug)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True,
              help="Configuration file path")
@click.option("--dataset", "-d", required=True, help="Dataset name to scan")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--fail-on-error", is_flag=True, help="Exit with error code on failed checks")
@click.pass_context
def scan(ctx: click.Context, config: str, dataset: str, 
         output: Optional[str], fail_on_error: bool) -> None:
    """Run data quality scan on a dataset."""
    console.print(f"[bold blue]DataSentinel DQ Scan[/bold blue]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Config: {config}")
    
    try:
        # Load configuration
        config_obj = load_config(Path(config))
        dataset_config = config_obj.get_dataset(dataset)
        
        if not dataset_config:
            console.print(f"[red]Dataset '{dataset}' not found in config[/red]")
            sys.exit(1)
        
        # Get adapter
        adapter_class = adapter_registry.get(dataset_config.storage_type)
        if not adapter_class:
            console.print(f"[red]Unknown storage type: {dataset_config.storage_type}[/red]")
            sys.exit(1)
        
        # Create dataset and adapter
        dataset_obj = Dataset(
            name=dataset_config.name,
            storage_type=dataset_config.storage_type,
            location=dataset_config.location,
            owner=dataset_config.owner,
            description=dataset_config.description,
            tags=dataset_config.tags
        )
        
        adapter = adapter_class()
        adapter.connect()
        
        # Create checks (simplified for demo)
        checks = []
        if dataset_config.checks:
            for i, check_name in enumerate(dataset_config.checks):
                check = Check(
                    name=check_name,
                    type="not_null",  # Simplified
                    severity=CheckSeverity.MEDIUM,
                    dimension=DQDimension.ACCURACY,
                    parameters={"columns": ["customer_id" if "customer" in check_name else "id"]}
                )
                checks.append(check)
        else:
            # Default check if none specified
            check = Check(
                name="default_not_null",
                type="not_null",
                severity=CheckSeverity.MEDIUM,
                dimension=DQDimension.ACCURACY,
                parameters={"columns": ["id"]}
            )
            checks.append(check)
        
        console.print(f"[yellow]Running {len(checks)} checks...[/yellow]")
        
        # Run validation
        results = adapter.validate(dataset_obj, checks)
        
        # Display results
        _display_results(results)
        
        # Save results if requested
        if output:
            with open(output, "w") as f:
                json.dump([r.model_dump() for r in results], f, indent=2, default=str)
            console.print(f"[green]Results saved to {output}[/green]")
        
        # Exit with error if requested and checks failed
        failed_checks = [r for r in results if r.status.value == "failed"]
        if fail_on_error and failed_checks:
            console.print(f"[red]{len(failed_checks)} checks failed[/red]")
            sys.exit(1)
        
        console.print(f"[green]Scan completed successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if ctx.obj.get("debug"):
            raise
        sys.exit(1)


@cli.command()
@click.option("--dataset", "-d", required=True, help="Dataset name")
@click.option("--format", "output_format", default="table", 
              type=click.Choice(["table", "json"]), help="Output format")
def score(dataset: str, output_format: str) -> None:
    """Display DQ score for a dataset."""
    console.print(f"[bold blue]DQ Score for {dataset}[/bold blue]")
    
    # For demo purposes, return a mock score
    mock_score = {
        "dataset_name": dataset,
        "overall_score": 0.85,
        "grade": "B+",
        "dimensions": {
            "accuracy": 0.90,
            "reliability": 0.80,
            "stewardship": 0.75,
            "usability": 0.85
        },
        "passed_checks": 15,
        "failed_checks": 3,
        "total_checks": 18
    }
    
    if output_format == "json":
        console.print(json.dumps(mock_score, indent=2))
    else:
        table = Table(title=f"DQ Score: {dataset}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Overall Score", f"{mock_score['overall_score']:.2f}")
        table.add_row("Grade", mock_score['grade'])
        table.add_row("Passed Checks", str(mock_score['passed_checks']))
        table.add_row("Failed Checks", str(mock_score['failed_checks']))
        
        console.print(table)
        
        # Dimensions table
        dim_table = Table(title="Dimension Scores")
        dim_table.add_column("Dimension", style="cyan")
        dim_table.add_column("Score", style="magenta")
        
        for dim, score in mock_score['dimensions'].items():
            dim_table.add_row(dim.title(), f"{score:.2f}")
        
        console.print(dim_table)


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
def serve(host: str, port: int) -> None:
    """Start the DataSentinel API server."""
    console.print(f"[bold blue]Starting DataSentinel API[/bold blue]")
    console.print(f"Server: http://{host}:{port}")
    console.print(f"Dashboard: http://{host}:8501")
    
    try:
        import uvicorn
        from dq_core.api.server import app
        
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        console.print("[red]uvicorn not available[/red]")
        sys.exit(1)


@cli.command()
def demo() -> None:
    """Run demo with sample data."""
    console.print("[bold blue]DataSentinel Demo[/bold blue]")
    console.print("This would run a demonstration with sample data...")
    console.print("[green]Demo completed![/green]")


def _display_results(results) -> None:
    """Display check results in a table."""
    table = Table(title="Data Quality Check Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Pass Rate", style="green")
    table.add_column("Records", style="yellow")
    
    for result in results:
        status_color = "green" if result.status.value == "passed" else "red"
        status = f"[{status_color}]{result.status.value.upper()}[/{status_color}]"
        
        pass_rate = f"{result.pass_rate:.2%}" if result.pass_rate is not None else "N/A"
        records = f"{result.passed_records}/{result.total_records}" if result.total_records else "N/A"
        
        table.add_row(result.check_name, status, pass_rate, records)
    
    console.print(table)


if __name__ == "__main__":
    cli()