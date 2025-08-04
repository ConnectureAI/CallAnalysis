"""
Command-line interface for the call analysis system.

This module provides CLI commands for administration, batch processing,
and system management tasks.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.panel import Panel
import pandas as pd

from .config import get_settings, reload_settings
from .database import init_db, cleanup_db, create_tables, drop_tables, check_database_health
from .analyzer import create_analyzer
from .predictor import create_predictor
from .coaching import create_coaching_system
from .models import CallInsight

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(config: Optional[str], verbose: bool):
    """Call Analysis System CLI - AI-powered dental call center analytics."""
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    if config:
        # In a real implementation, you'd load custom config here
        pass
    
    settings = get_settings()
    if verbose:
        console.print(f"[green]Configuration loaded[/green]")
        console.print(f"Environment: {settings.environment}")
        console.print(f"Debug mode: {settings.debug}")


@cli.group()
def db():
    """Database management commands."""
    pass


@db.command()
def init():
    """Initialize the database with tables and initial data."""
    async def _init():
        try:
            await init_db()
            console.print("[green]✓[/green] Database initialized successfully")
        except Exception as e:
            console.print(f"[red]✗[/red] Database initialization failed: {e}")
            sys.exit(1)
    
    asyncio.run(_init())


@db.command()
def reset():
    """Reset the database (drop and recreate all tables)."""
    
    if not click.confirm("This will delete all data. Are you sure?"):
        console.print("Operation cancelled")
        return
    
    async def _reset():
        try:
            console.print("Dropping tables...")
            await drop_tables()
            
            console.print("Creating tables...")
            await create_tables()
            
            console.print("[green]✓[/green] Database reset successfully")
        except Exception as e:
            console.print(f"[red]✗[/red] Database reset failed: {e}")
            sys.exit(1)
    
    asyncio.run(_reset())


@db.command()
def health():
    """Check database health."""
    async def _health():
        try:
            health_info = await check_database_health()
            
            table = Table(title="Database Health Check")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")
            
            status_color = "green" if health_info["status"] == "healthy" else "red"
            table.add_row("Overall Status", f"[{status_color}]{health_info['status']}[/{status_color}]", "")
            
            if "database_version" in health_info:
                table.add_row("Database Version", health_info["database_version"], "")
            
            if "active_connections" in health_info:
                table.add_row("Active Connections", str(health_info["active_connections"]), "")
            
            if "engine_pool_size" in health_info:
                table.add_row("Pool Size", str(health_info["engine_pool_size"]), "")
                table.add_row("Checked Out", str(health_info["engine_checked_out"]), "")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]✗[/red] Health check failed: {e}")
            sys.exit(1)
    
    asyncio.run(_health())


@cli.group()
def analyze():
    """Call analysis commands."""
    pass


@analyze.command()
@click.option('--file', '-f', required=True, help='Path to file containing call data')
@click.option('--format', '-fmt', default='json', type=click.Choice(['json', 'csv']), help='Input file format')
@click.option('--output', '-o', help='Output file path for results')
@click.option('--batch-size', '-b', default=10, help='Batch size for processing')
def batch(file: str, format: str, output: Optional[str], batch_size: int):
    """Analyze calls from a batch file."""
    
    input_path = Path(file)
    if not input_path.exists():
        console.print(f"[red]✗[/red] File not found: {file}")
        sys.exit(1)
    
    async def _batch_analyze():
        try:
            # Load input data
            console.print(f"Loading data from {file}...")
            
            if format == 'json':
                with open(input_path, 'r') as f:
                    data = json.load(f)
            elif format == 'csv':
                df = pd.read_csv(input_path)
                data = df.to_dict('records')
            
            if not data:
                console.print("[red]✗[/red] No data found in input file")
                return
            
            # Validate data format
            required_fields = ['call_id', 'transcript']
            for i, item in enumerate(data):
                if not all(field in item for field in required_fields):
                    console.print(f"[red]✗[/red] Missing required fields in record {i}: {required_fields}")
                    return
            
            # Initialize analyzer
            console.print("Initializing analyzer...")
            analyzer = create_analyzer()
            
            # Validate API key
            if not await analyzer.validate_api_key():
                console.print("[red]✗[/red] Invalid OpenAI API key")
                return
            
            # Process in batches
            results = []
            total_items = len(data)
            
            with Progress() as progress:
                task = progress.add_task("Analyzing calls...", total=total_items)
                
                for i in range(0, total_items, batch_size):
                    batch_data = data[i:i + batch_size]
                    
                    try:
                        batch_results = await analyzer.analyze_batch(batch_data)
                        results.extend([insight.to_dict() for insight in batch_results])
                        
                        progress.update(task, advance=len(batch_data))
                        
                    except Exception as e:
                        console.print(f"[red]✗[/red] Error processing batch {i//batch_size + 1}: {e}")
                        continue
            
            # Save results
            if output:
                output_path = Path(output)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                console.print(f"[green]✓[/green] Results saved to {output}")
            
            # Print summary
            console.print(f"\n[green]✓[/green] Analysis completed:")
            console.print(f"  Total calls processed: {len(results)}")
            
            if results:
                avg_confidence = sum(r['confidence_score'] for r in results) / len(results)
                avg_sentiment = sum(r['sentiment_score'] for r in results) / len(results)
                
                console.print(f"  Average confidence: {avg_confidence:.3f}")
                console.print(f"  Average sentiment: {avg_sentiment:.3f}")
                
                # Top intents
                intents = [r['primary_intent'] for r in results]
                intent_counts = {}
                for intent in intents:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
                
                console.print("\n  Top intents:")
                for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    console.print(f"    {intent}: {count}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Batch analysis failed: {e}")
            sys.exit(1)
    
    asyncio.run(_batch_analyze())


@analyze.command()
@click.argument('call_id')
@click.argument('transcript')
def single(call_id: str, transcript: str):
    """Analyze a single call transcript."""
    
    async def _single_analyze():
        try:
            # Initialize analyzer
            analyzer = create_analyzer()
            
            # Validate API key
            if not await analyzer.validate_api_key():
                console.print("[red]✗[/red] Invalid OpenAI API key")
                return
            
            # Analyze transcript
            console.print(f"Analyzing call {call_id}...")
            insight = await analyzer.analyze_transcript(transcript, call_id)
            
            # Display results
            panel = Panel.fit(
                f"""[bold]Call Analysis Results[/bold]
                
Call ID: {insight.call_id}
Primary Intent: {insight.primary_intent}
Sentiment Score: {insight.sentiment_score:.3f}
Urgency Level: {insight.urgency_level}/5
Confidence: {insight.confidence_score:.3f}
Resolution Status: {insight.resolution_status}

Revenue Opportunity: ${insight.revenue_opportunity:.2f}

Service Requests: {', '.join(insight.service_requests) if insight.service_requests else 'None'}
Pain Points: {', '.join(insight.pain_points) if insight.pain_points else 'None'}
Success Factors: {', '.join(insight.success_factors) if insight.success_factors else 'None'}

Next Actions: {', '.join(insight.next_actions) if insight.next_actions else 'None'}""",
                title="Analysis Results",
                border_style="green"
            )
            
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]✗[/red] Analysis failed: {e}")
            sys.exit(1)
    
    asyncio.run(_single_analyze())


@cli.group()
def models():
    """Machine learning model management."""
    pass


@models.command()
def train():
    """Train predictive models."""
    
    async def _train():
        try:
            console.print("Initializing predictor...")
            predictor = create_predictor()
            
            console.print("Training models...")
            results = predictor.train_models()
            
            if not results:
                console.print("[yellow]⚠[/yellow] Insufficient data for training")
                console.print(f"Need at least {predictor.min_training_samples} samples")
                console.print(f"Current samples: {len(predictor.historical_insights)}")
                return
            
            # Display training results
            table = Table(title="Model Training Results")
            table.add_column("Model", style="cyan")
            table.add_column("MAE", style="green")
            table.add_column("R²", style="green")
            
            for model_name, metrics in results.items():
                mae = metrics.get('mae', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                
                mae_str = f"{mae:.3f}" if isinstance(mae, (int, float)) else str(mae)
                r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)
                
                table.add_row(model_name, mae_str, r2_str)
            
            console.print(table)
            console.print("[green]✓[/green] Model training completed")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Model training failed: {e}")
            sys.exit(1)
    
    asyncio.run(_train())


@models.command()
def info():
    """Display model information."""
    
    async def _info():
        try:
            predictor = create_predictor()
            info = predictor.get_model_info()
            
            table = Table(title="Model Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Trained", "Yes" if info['is_trained'] else "No")
            table.add_row("Model Count", str(info['model_count']))
            table.add_row("Training Samples", str(info['training_samples']))
            table.add_row("Models Directory", info['models_dir'])
            
            if info['models_available']:
                table.add_row("Available Models", ", ".join(info['models_available']))
            
            console.print(table)
            
            # Model metadata
            if info.get('model_metadata'):
                console.print("\n[bold]Model Details:[/bold]")
                for model_name, metadata in info['model_metadata'].items():
                    console.print(f"\n[cyan]{model_name}:[/cyan]")
                    for key, value in metadata.items():
                        console.print(f"  {key}: {value}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Error getting model info: {e}")
            sys.exit(1)
    
    asyncio.run(_info())


@models.command()
@click.option('--days', '-d', default=7, help='Number of days to predict')
def predict(days: int):
    """Generate predictions."""
    
    async def _predict():
        try:
            predictor = create_predictor()
            
            if not predictor.is_trained:
                console.print("[red]✗[/red] Models not trained. Run 'models train' first.")
                return
            
            console.print(f"Generating predictions for {days} days...")
            predictions = predictor.predict_next_period(days)
            
            if 'error' in predictions:
                console.print(f"[red]✗[/red] {predictions['error']}")
                return
            
            # Display daily forecasts
            if predictions.get('daily_forecasts'):
                table = Table(title="Daily Forecasts")
                table.add_column("Date", style="cyan")
                table.add_column("Calls", style="green")
                table.add_column("Sentiment", style="green")
                table.add_column("Revenue", style="green")
                
                for forecast in predictions['daily_forecasts']:
                    table.add_row(
                        forecast.get('date', 'N/A'),
                        str(forecast.get('call_volume', 0)),
                        f"{forecast.get('sentiment', 0):.3f}",
                        f"${forecast.get('revenue', 0):.0f}"
                    )
                
                console.print(table)
            
            # Display weekly totals
            if predictions.get('weekly_totals'):
                totals = predictions['weekly_totals']
                console.print(f"\n[bold]Weekly Totals:[/bold]")
                console.print(f"  Total Calls: {totals.get('total_calls', 0)}")
                console.print(f"  Avg Sentiment: {totals.get('avg_sentiment', 0):.3f}")
                console.print(f"  Total Revenue Opportunity: ${totals.get('total_revenue_opportunity', 0):,.0f}")
            
            # Display recommendations
            if predictions.get('recommendations'):
                console.print(f"\n[bold]Recommendations:[/bold]")
                for i, rec in enumerate(predictions['recommendations'], 1):
                    console.print(f"  {i}. {rec}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Prediction failed: {e}")
            sys.exit(1)
    
    asyncio.run(_predict())


@cli.group()
def server():
    """Server management commands."""
    pass


@server.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development only)')
def start(host: str, port: int, workers: int, reload: bool):
    """Start the FastAPI server."""
    import uvicorn
    
    console.print(f"Starting Call Analysis API server...")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Workers: {workers}")
    console.print(f"Reload: {reload}")
    
    try:
        uvicorn.run(
            "call_analysis.api:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]✗[/red] Server failed to start: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show system status."""
    settings = get_settings()
    
    # System information
    table = Table(title="Call Analysis System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    table.add_row("Environment", settings.environment, "")
    table.add_row("Debug Mode", "Enabled" if settings.debug else "Disabled", "")
    table.add_row("Version", settings.app_version, "")
    
    # Feature flags
    table.add_row("Semantic Analysis", "Enabled" if settings.enable_semantic_analysis else "Disabled", "")
    table.add_row("Predictive Analytics", "Enabled" if settings.enable_predictive_analytics else "Disabled", "")
    table.add_row("Real-time Coaching", "Enabled" if settings.enable_real_time_coaching else "Disabled", "")
    
    console.print(table)
    
    # Database status check
    async def _check_db():
        try:
            health = await check_database_health()
            status_color = "green" if health["status"] == "healthy" else "red"
            console.print(f"\nDatabase: [{status_color}]{health['status']}[/{status_color}]")
        except Exception as e:
            console.print(f"\nDatabase: [red]Error - {e}[/red]")
    
    asyncio.run(_check_db())


@cli.command()
@click.option('--export-format', default='json', type=click.Choice(['json', 'csv']), help='Export format')
@click.option('--output', '-o', help='Output file path')
@click.option('--days', '-d', default=30, help='Number of days to export')
def export(export_format: str, output: Optional[str], days: int):
    """Export call insights data."""
    
    async def _export():
        try:
            from .database import get_db_session
            from .models import CallInsightDB
            from sqlalchemy import select
            from datetime import datetime, timedelta
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            console.print(f"Exporting data from {start_date.date()} to {end_date.date()}...")
            
            async with get_db_session() as session:
                stmt = select(CallInsightDB).where(
                    CallInsightDB.timestamp >= start_date,
                    CallInsightDB.timestamp <= end_date
                ).order_by(CallInsightDB.timestamp.desc())
                
                result = await session.execute(stmt)
                insights = result.scalars().all()
            
            if not insights:
                console.print("[yellow]⚠[/yellow] No data found for the specified period")
                return
            
            # Convert to exportable format
            data = []
            for insight in insights:
                data.append({
                    'call_id': insight.call_id,
                    'timestamp': insight.timestamp.isoformat(),
                    'clinic_mentioned': insight.clinic_mentioned,
                    'primary_intent': insight.primary_intent,
                    'secondary_intents': insight.secondary_intents,
                    'sentiment_score': insight.sentiment_score,
                    'urgency_level': insight.urgency_level,
                    'resolution_status': insight.resolution_status,
                    'revenue_opportunity': insight.revenue_opportunity,
                    'service_requests': insight.service_requests,
                    'pain_points': insight.pain_points,
                    'success_factors': insight.success_factors,
                    'next_actions': insight.next_actions,
                    'confidence_score': insight.confidence_score,
                })
            
            # Export data
            if not output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = f"call_insights_{timestamp}.{export_format}"
            
            output_path = Path(output)
            
            if export_format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif export_format == 'csv':
                df = pd.DataFrame(data)
                # Flatten list columns
                for col in ['secondary_intents', 'service_requests', 'pain_points', 'success_factors', 'next_actions']:
                    df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                df.to_csv(output_path, index=False)
            
            console.print(f"[green]✓[/green] Exported {len(data)} records to {output_path}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Export failed: {e}")
            sys.exit(1)
    
    asyncio.run(_export())


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()