"""
Server module for running the Call Analysis application.

This module provides utilities for starting and managing the FastAPI server
with proper configuration and lifecycle management.
"""

import logging
import logging.config
import sys
from typing import Optional

import uvicorn
from rich.console import Console

from .config import get_settings
from .api import app

console = Console()
logger = logging.getLogger(__name__)


def configure_logging(settings):
    """Configure application logging."""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "json": {
                "()": "structlog.stdlib.ProcessorFormatter",
                "processor": "structlog.dev.ConsoleRenderer" if settings.debug else "structlog.processors.JSONRenderer",
            },
        },
        "handlers": {
            "default": {
                "formatter": "json" if settings.logging.format == "json" else "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.logging.level,
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {"level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"level": "INFO"},
            "call_analysis": {"level": settings.logging.level},
            "sqlalchemy.engine": {"level": "WARNING"},
        },
    }
    
    # Add file handler if specified
    if settings.logging.file:
        log_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": settings.logging.file,
            "maxBytes": settings.logging.max_size,
            "backupCount": settings.logging.backup_count,
            "formatter": "json" if settings.logging.format == "json" else "default",
        }
        log_config["root"]["handlers"].append("file")
    
    logging.config.dictConfig(log_config)


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    reload: Optional[bool] = None,
    log_level: Optional[str] = None,
):
    """
    Run the FastAPI server with proper configuration.
    
    Args:
        host: Host to bind to (overrides config)
        port: Port to bind to (overrides config)
        workers: Number of worker processes (overrides config)
        reload: Enable auto-reload (overrides config)
        log_level: Log level (overrides config)
    """
    settings = get_settings()
    
    # Use provided values or fall back to settings
    host = host or settings.host
    port = port or settings.port
    workers = workers or settings.workers
    reload = reload if reload is not None else settings.debug
    log_level = log_level or settings.logging.level.lower()
    
    # Configure logging
    configure_logging(settings)
    
    # Display startup information
    console.print(f"[bold green]🚀 Starting Call Analysis API Server[/bold green]")
    console.print(f"Environment: [cyan]{settings.environment}[/cyan]")
    console.print(f"Version: [cyan]{settings.app_version}[/cyan]")
    console.print(f"Host: [cyan]{host}[/cyan]")
    console.print(f"Port: [cyan]{port}[/cyan]")
    console.print(f"Workers: [cyan]{workers if not reload else 1}[/cyan]")
    console.print(f"Reload: [cyan]{reload}[/cyan]")
    console.print(f"Debug: [cyan]{settings.debug}[/cyan]")
    
    if settings.debug:
        console.print(f"📚 API Documentation: [link]http://{host}:{port}/docs[/link]")
        console.print(f"📖 ReDoc Documentation: [link]http://{host}:{port}/redoc[/link]")
    
    # Validate configuration
    try:
        # Check OpenAI API key
        if not settings.openai.api_key or settings.openai.api_key == "":
            console.print("[yellow]⚠️  Warning: OpenAI API key not configured. AI features will be disabled.[/yellow]")
        
        # Check database configuration
        if not settings.database.host or not settings.database.name:
            console.print("[red]❌ Error: Database configuration incomplete.[/red]")
            sys.exit(1)
        
        logger.info("Server configuration validated successfully")
        
    except Exception as e:
        console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
        sys.exit(1)
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "call_analysis.api:app",
        "host": host,
        "port": port,
        "log_level": log_level,
        "access_log": True,
        "use_colors": True,
        "loop": "asyncio",
    }
    
    # Set workers (only if not in reload mode)
    if reload:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["src"]
    else:
        uvicorn_config["workers"] = workers
    
    # Production optimizations
    if settings.is_production:
        uvicorn_config.update({
            "access_log": False,  # Disable access logs in production for performance
            "server_header": False,  # Don't expose server information
            "date_header": False,  # Don't add date header
        })
    
    try:
        # Start the server
        logger.info("Starting uvicorn server", config=uvicorn_config)
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Server stopped by user[/yellow]")
        logger.info("Server stopped by user interrupt")
        
    except Exception as e:
        console.print(f"[red]❌ Server failed to start: {e}[/red]")
        logger.error("Server startup failed", error=str(e), exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Call Analysis API Server")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default=None, help="Log level")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()