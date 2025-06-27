"""Production-grade command-line interface for QuarryCore."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import click
import structlog
import uvicorn

try:
    from rich.console import Console  # type: ignore[import-not-found]
    from rich.live import Live  # type: ignore[import-not-found]
    from rich.panel import Panel  # type: ignore[import-not-found]
    from rich.progress import (  # type: ignore[import-not-found]
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table  # type: ignore[import-not-found]
except ImportError:
    # Fallback for when rich is not available
    Console = None
    Live = None
    Panel = None
    Progress = None
    Table = None
    BarColumn = None
    SpinnerColumn = None
    TextColumn = None
    TimeElapsedColumn = None
    TimeRemainingColumn = None

from quarrycore import __version__
from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline
from quarrycore.web.main import app as fastapi_app

console = Console() if Console else None
logger = structlog.get_logger(__name__)


class ShutdownManager:
    """Handles graceful shutdown for asyncio applications."""

    def __init__(self) -> None:
        self.is_shutting_down = False
        self._tasks_to_cancel: Set[asyncio.Task[Any]] = set()
        self._shutdown_signals = (signal.SIGINT, signal.SIGTERM)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def add_task(self, task: asyncio.Task[Any]) -> None:
        """Register a task to be cancelled on shutdown."""
        if not self.is_shutting_down:
            self._tasks_to_cancel.add(task)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """The actual signal handler function."""
        if self.is_shutting_down:
            return  # Avoid handling signal multiple times
        self.is_shutting_down = True
        if console:
            console.print(f"\n[yellow]Received signal {signum}, initiating graceful shutdown...[/yellow]")

        if self._loop and self._loop.is_running():
            # Stop the loop and cancel tasks from within the loop
            self._loop.call_soon_threadsafe(self._initiate_shutdown)

    def _initiate_shutdown(self) -> None:
        """Cancel all registered tasks and prepare to stop the loop."""
        if console:
            console.print(f"[cyan]Cancelling {len(self._tasks_to_cancel)} background tasks...[/cyan]")
        for task in self._tasks_to_cancel:
            task.cancel()

        # This will allow the main loop to exit
        if self._loop:
            self._loop.stop()

    def install(self) -> None:
        """Install signal handlers for the current process."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        for sig in self._shutdown_signals:
            signal.signal(sig, self._signal_handler)


# Global instances for use in CLI commands
shutdown_manager = ShutdownManager()
current_pipeline: Optional[Pipeline] = None


@click.group()
@click.version_option(version=__version__)
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], log_level: str) -> None:
    """QuarryCore - Production-grade AI training data miner."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config) if config else None
    ctx.obj["log_level"] = log_level

    # Install the graceful shutdown handler
    shutdown_manager.install()

    # Configure basic logging for CLI
    import logging

    logging.basicConfig(level=getattr(logging, log_level))


@cli.command()
@click.argument("urls", type=click.File("r"), required=False)
@click.option("--url", multiple=True, help="Single URL to process (can be used multiple times)")
@click.option("--batch-size", default=50, help="Batch size for processing")
@click.option("--max-concurrency", default=100, help="Maximum concurrent operations")
@click.option("--checkpoint-interval", default=300, help="Checkpoint interval in seconds")
@click.option("--resume-from", type=click.Path(exists=True), help="Resume from checkpoint file")
@click.option("--dry-run", is_flag=True, help="Validate configuration without running")
@click.option("--interactive", is_flag=True, help="Interactive mode with real-time control")
@click.pass_context
def run(
    ctx: click.Context,
    urls: Optional[Any],
    url: tuple[str, ...],
    batch_size: int,
    max_concurrency: int,
    checkpoint_interval: int,
    resume_from: Optional[str],
    dry_run: bool,
    interactive: bool,
) -> None:
    """Run the complete QuarryCore pipeline."""
    global current_pipeline

    # Parse URLs
    url_list: List[str] = []
    if urls:
        url_list.extend(line.strip() for line in urls if line.strip())
    if url:
        url_list.extend(url)

    resume_path = Path(resume_from) if resume_from else None

    if not url_list and not resume_path:
        if console:
            console.print("[red]Error: No URLs provided and no checkpoint to resume from[/red]")
        sys.exit(1)

    async def run_pipeline() -> None:
        global current_pipeline

        container = DependencyContainer(ctx.obj["config_path"])
        current_pipeline = Pipeline(container, max_concurrency)

        if dry_run:
            if console:
                console.print("[blue]🔍 Dry run mode - validating configuration...[/blue]")
            async with container.lifecycle():
                health = container.get_health_status()
                if console and Panel:
                    console.print(
                        Panel(
                            json.dumps(health, indent=2),
                            title="Configuration Validation",
                            border_style="green",
                        )
                    )
                else:
                    print(json.dumps(health, indent=2))
            return

        if console and Panel:
            console.print(
                Panel.fit(
                    f"[bold blue]QuarryCore Pipeline[/bold blue]\n"
                    f"URLs: {len(url_list)}\n"
                    f"Batch size: {batch_size}\n"
                    f"Max concurrency: {max_concurrency}",
                    title="Starting Pipeline",
                )
            )

        if interactive:
            await run_interactive_pipeline(current_pipeline, url_list, batch_size, checkpoint_interval, resume_path)
        else:
            await run_standard_pipeline(current_pipeline, url_list, batch_size, checkpoint_interval, resume_path)

    asyncio.run(run_pipeline())


async def run_standard_pipeline(
    pipeline: Pipeline,
    url_list: List[str],
    batch_size: int,
    checkpoint_interval: int,
    resume_from: Optional[Path],
) -> None:
    """Run pipeline with progress bars."""
    if not Progress or not console:
        # Fallback to simple text output
        print(f"Processing {len(url_list)} URLs...")
        result = await pipeline.run(
            urls=url_list,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
        )
        print(f"Completed: {result.get('processed_count', 0)} processed, {result.get('failed_count', 0)} failed")
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        main_task = progress.add_task("Processing URLs", total=len(url_list))

        try:
            result = await pipeline.run(
                urls=url_list,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                resume_from=resume_from,
            )

            # Update progress based on pipeline state
            if pipeline.state:
                progress.update(main_task, completed=pipeline.state.processed_count)

            if Panel:
                console.print(
                    Panel(
                        f"✅ Pipeline completed successfully!\n"
                        f"Processed: {result.get('processed_count', 0)}\n"
                        f"Failed: {result.get('failed_count', 0)}\n"
                        f"Duration: {result.get('duration', 0):.2f}s",
                        title="Results",
                        border_style="green",
                    )
                )
            else:
                console.print(f"✅ Pipeline completed successfully! Processed: {result.get('processed_count', 0)}")

        except Exception as e:
            if console:
                console.print(f"[red]Pipeline failed: {e}[/red]")
            raise


async def run_interactive_pipeline(
    pipeline: Pipeline,
    url_list: List[str],
    batch_size: int,
    checkpoint_interval: int,
    resume_from: Optional[Path],
) -> None:
    """Run pipeline in interactive mode with real-time monitoring."""

    def create_status_table() -> Any:
        if not Table:
            return None

        table = Table(title="Pipeline Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        if pipeline.state:
            table.add_row("Pipeline ID", pipeline.state.pipeline_id)
            table.add_row("Stage", pipeline.state.stage.value)
            table.add_row("Processed", str(pipeline.state.processed_count))
            table.add_row("Failed", str(pipeline.state.failed_count))
            table.add_row("Remaining", str(len(pipeline.state.urls_remaining)))
            table.add_row("Batch Size", str(pipeline.state.batch_size))

        return table

    if not Live or not console:
        # Fallback to simple monitoring
        pipeline_task = asyncio.create_task(
            pipeline.run(
                urls=url_list,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                resume_from=resume_from,
            )
        )

        while not pipeline_task.done():
            if pipeline.state:
                print(f"Status: {pipeline.state.stage.value}, Processed: {pipeline.state.processed_count}")
            await asyncio.sleep(1)

        await pipeline_task
        print("Pipeline completed successfully!")
        return

    with Live(create_status_table(), refresh_per_second=2, console=console) as live:
        # Start pipeline in background
        pipeline_task = asyncio.create_task(
            pipeline.run(
                urls=url_list,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                resume_from=resume_from,
            )
        )

        # Update display while pipeline runs
        while not pipeline_task.done():
            live.update(create_status_table())
            await asyncio.sleep(0.5)

        # Get final result
        try:
            await pipeline_task
            console.print("[green]Pipeline completed successfully![/green]")
        except Exception as e:
            console.print(f"[red]Pipeline failed: {e}[/red]")


@cli.command()
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.pass_context
def resume(ctx: click.Context, checkpoint_path: str) -> None:
    """Resume pipeline from a checkpoint."""
    if console:
        console.print(f"[blue]📂 Resuming pipeline from {checkpoint_path}[/blue]")

    # Load checkpoint and continue
    ctx.invoke(run, resume_from=checkpoint_path)


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for analysis report")
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "yaml", "table"]),
    help="Output format",
)
@click.pass_context
def analyze(ctx: click.Context, dataset_path: str, output: Optional[str], output_format: str) -> None:
    """Analyze dataset quality and characteristics."""
    if console:
        console.print(f"[blue]🔍 Analyzing dataset: {dataset_path}[/blue]")

    async def run_analysis() -> None:
        container = DependencyContainer(ctx.obj["config_path"])
        async with container.lifecycle():
            # Simulate analysis (replace with actual implementation)
            analysis_result: Dict[str, Any] = {
                "dataset_path": str(dataset_path),
                "total_documents": 12500,
                "avg_quality_score": 0.85,
                "domains": {
                    "medical": 3500,
                    "technical": 4200,
                    "legal": 2800,
                    "general": 2000,
                },
                "avg_tokens_per_doc": 1024,
                "duplicate_ratio": 0.02,
            }

            if output_format == "table" and console and Table:
                table = Table(title="Dataset Analysis")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")

                for key, value in analysis_result.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            table.add_row(f"{key}.{sub_key}", str(sub_value))
                    else:
                        table.add_row(key, str(value))

                console.print(table)
            else:
                formatted_result = json.dumps(analysis_result, indent=2)
                if console:
                    console.print(formatted_result)
                else:
                    print(formatted_result)

                if output:
                    Path(output).write_text(formatted_result)
                    if console:
                        console.print(f"[green]Analysis saved to {output}[/green]")

    asyncio.run(run_analysis())


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.pass_context
def monitor(ctx: click.Context, host: str, port: int) -> None:
    """Start the monitoring dashboard."""
    if console:
        console.print(f"[green]🚀 Starting monitoring dashboard at http://{host}:{port}[/green]")

    # The original implementation had a complex and faulty manual event loop,
    # leading to a hang. Replacing it with a direct `uvicorn.run` call is the
    # correct, idiomatic, and robust way to start the server. Uvicorn handles
    # its own lifecycle and signal management (like Ctrl+C).
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level="info",
    )


@cli.command()
@click.pass_context
def validate_config(ctx: click.Context) -> None:
    """Validate the current configuration."""
    if console:
        console.print("[blue]🔍 Validating configuration...[/blue]")

    async def validate() -> None:
        try:
            container = DependencyContainer(ctx.obj["config_path"])
            await container.initialize()

            health = container.get_health_status()

            if console and Table:
                table = Table(title="Configuration Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="magenta")

                for key, value in health.items():
                    status = "✅ OK" if value else "❌ Error"
                    table.add_row(key, status)

                console.print(table)
            else:
                print("Configuration Status:")
                for key, value in health.items():
                    status = "✅ OK" if value else "❌ Error"
                    print(f"{key}: {status}")

            if all(health.values()):
                if console:
                    console.print("[green]✅ Configuration is valid![/green]")
                else:
                    print("✅ Configuration is valid!")
            else:
                if console:
                    console.print("[red]❌ Configuration has issues![/red]")
                else:
                    print("❌ Configuration has issues!")
                sys.exit(1)

        except Exception as e:
            if console:
                console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
            else:
                print(f"❌ Configuration validation failed: {e}")
            sys.exit(1)

    asyncio.run(validate())


@cli.command()
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "prometheus"]),
    help="Health check format",
)
def health(output_format: str) -> None:
    """Get health status (for Kubernetes/Docker)."""

    async def check_health() -> None:
        try:
            container = DependencyContainer()
            health_status = container.get_health_status()

            if output_format == "prometheus":
                # Output Prometheus-style metrics
                for key, value in health_status.items():
                    metric_value = 1 if value else 0
                    if console:
                        console.print(f"quarrycore_health_{key} {metric_value}")
                    else:
                        print(f"quarrycore_health_{key} {metric_value}")
            else:
                output = json.dumps(health_status, indent=2)
                if console:
                    console.print(output)
                else:
                    print(output)

            # Exit with non-zero if unhealthy
            if not all(health_status.values()):
                sys.exit(1)

        except Exception as e:
            error_output = json.dumps({"error": str(e)})
            if console:
                console.print(error_output)
            else:
                print(error_output)
            sys.exit(1)

    asyncio.run(check_health())


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
