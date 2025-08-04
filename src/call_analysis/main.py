#!/usr/bin/env python3
"""
Main entry point for Call Analysis System.

Provides command-line interface for all system components including
analytics, AI agents, data management, and system administration.
"""

import click
from .cli import agents


@click.group()
@click.version_option()
def cli():
    """
    Call Analysis System - AI-powered call center analytics platform.
    
    This system provides advanced analytics, automated call processing,
    and intelligent insights for call center operations.
    """
    pass


# Add command groups
cli.add_command(agents)


@cli.command()
def info():
    """Show system information."""
    click.echo("🤖 Call Analysis System")
    click.echo("=" * 30)
    click.echo("AI-powered call center analytics platform")
    click.echo()
    click.echo("Features:")
    click.echo("• Advanced NLP analysis (sentiment, entities, topics)")
    click.echo("• Automated call acquisition and processing")
    click.echo("• Intelligent workflow orchestration")
    click.echo("• Real-time system monitoring")
    click.echo("• Comprehensive analytics and reporting")
    click.echo()
    click.echo("Commands:")
    click.echo("• agents start    - Start the AI agent system")
    click.echo("• agents status   - Check system status")
    click.echo("• agents test     - Run system tests")


if __name__ == "__main__":
    cli()