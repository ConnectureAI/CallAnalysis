"""
CLI commands for managing the AI agent system.

Provides command-line interface for starting, stopping, monitoring,
and managing the AI agent system components.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional
import json

import click
from ..agents import AgentManager
from ..config import get_settings

logger = logging.getLogger(__name__)


class AgentSystemCLI:
    """CLI interface for the agent system."""
    
    def __init__(self):
        self.agent_manager: Optional[AgentManager] = None
        self.shutdown_event = asyncio.Event()
    
    async def start_system(self) -> None:
        """Start the agent system."""
        try:
            self.agent_manager = AgentManager()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Initialize and start the system
            await self.agent_manager.initialize()
            
            click.echo("✅ AI Agent System started successfully!")
            click.echo("\nRegistered agents:")
            
            for agent_id in self.agent_manager.get_registered_agents():
                status = "🟢 Running" if self.agent_manager.is_agent_running(agent_id) else "🔴 Stopped"
                click.echo(f"  • {agent_id}: {status}")
            
            click.echo("\nPress Ctrl+C to shutdown the system...")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            click.echo(f"❌ Error starting agent system: {e}")
            logger.error(f"Failed to start agent system: {e}")
            raise
        
        finally:
            if self.agent_manager:
                await self.agent_manager.shutdown()
                click.echo("🛑 Agent system shutdown complete")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            click.echo("\n🛑 Shutdown signal received...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def get_system_status(self) -> dict:
        """Get system status information."""
        if not self.agent_manager:
            self.agent_manager = AgentManager()
            await self.agent_manager.initialize()
        
        return await self.agent_manager.get_system_health()
    
    async def get_agent_status(self, agent_id: str) -> dict:
        """Get status of a specific agent."""
        if not self.agent_manager:
            return {"error": "Agent system not initialized"}
        
        return await self.agent_manager.get_agent_status(agent_id)


@click.group()
def agents():
    """Manage AI agent system."""
    pass


@agents.command()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def start(debug):
    """Start the AI agent system."""
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the system
    cli = AgentSystemCLI()
    
    try:
        asyncio.run(cli.start_system())
    except KeyboardInterrupt:
        click.echo("\n🛑 Shutdown requested by user")
    except Exception as e:
        click.echo(f"❌ Fatal error: {e}")
        sys.exit(1)


@agents.command()
@click.option('--format', type=click.Choice(['json', 'table']), default='table', help='Output format')
def status(format):
    """Get system status."""
    
    async def get_status():
        cli = AgentSystemCLI()
        try:
            status_data = await cli.get_system_status()
            
            if format == 'json':
                click.echo(json.dumps(status_data, indent=2))
            else:
                # Table format
                click.echo("🤖 AI Agent System Status")
                click.echo("=" * 40)
                
                overall_status = status_data.get("overall_status", "unknown")
                status_icon = "🟢" if overall_status == "healthy" else "🟡" if overall_status == "degraded" else "🔴"
                click.echo(f"Overall Status: {status_icon} {overall_status.upper()}")
                
                system_info = status_data.get("system_info", {})
                click.echo(f"System Status: {system_info.get('status', 'unknown')}")
                click.echo(f"Uptime: {system_info.get('uptime', 0):.0f} seconds")
                click.echo(f"Agents Running: {system_info.get('agents_running', 0)}/{system_info.get('agents_registered', 0)}")
                
                # Agent details
                click.echo("\nAgent Details:")
                agents_data = status_data.get("agents", {})
                for agent_id, agent_status in agents_data.items():
                    status_name = agent_status.get("status", "unknown")
                    status_icon = "🟢" if status_name in ["idle", "running"] else "🟡" if status_name == "busy" else "🔴"
                    active_tasks = agent_status.get("active_tasks", 0)
                    queue_size = agent_status.get("queue_size", 0)
                    click.echo(f"  • {agent_id}: {status_icon} {status_name} (Tasks: {active_tasks}, Queue: {queue_size})")
                
                # Issues
                issues = status_data.get("issues", [])
                if issues:
                    click.echo("\n⚠️  Issues:")
                    for issue in issues:
                        click.echo(f"  • {issue}")
                
        except Exception as e:
            click.echo(f"❌ Error getting status: {e}")
            
        finally:
            if cli.agent_manager:
                await cli.agent_manager.shutdown()
    
    asyncio.run(get_status())


@agents.command()
@click.argument('agent_id')
@click.option('--format', type=click.Choice(['json', 'table']), default='table', help='Output format')
def agent_status(agent_id, format):
    """Get status of a specific agent."""
    
    async def get_agent_status():
        cli = AgentSystemCLI()
        try:
            status_data = await cli.get_agent_status(agent_id)
            
            if format == 'json':
                click.echo(json.dumps(status_data, indent=2))
            else:
                if "error" in status_data:
                    click.echo(f"❌ {status_data['error']}")
                    return
                
                click.echo(f"🤖 Agent: {agent_id}")
                click.echo("=" * 40)
                click.echo(f"Status: {status_data.get('status', 'unknown')}")
                click.echo(f"Active Tasks: {status_data.get('active_tasks', 0)}")
                click.echo(f"Queue Size: {status_data.get('queue_size', 0)}")
                click.echo(f"Uptime: {status_data.get('uptime', 0):.0f} seconds")
                
                metrics = status_data.get("metrics", {})
                if metrics:
                    click.echo("\nMetrics:")
                    click.echo(f"  • Tasks Completed: {metrics.get('tasks_completed', 0)}")
                    click.echo(f"  • Tasks Failed: {metrics.get('tasks_failed', 0)}")
                    click.echo(f"  • Messages Sent: {metrics.get('messages_sent', 0)}")
                    click.echo(f"  • Messages Received: {metrics.get('messages_received', 0)}")
                
        except Exception as e:
            click.echo(f"❌ Error getting agent status: {e}")
            
        finally:
            if cli.agent_manager:
                await cli.agent_manager.shutdown()
    
    asyncio.run(get_agent_status())


@agents.command()
def test():
    """Test the agent system with sample data."""
    click.echo("🧪 Testing AI Agent System...")
    
    async def run_test():
        cli = AgentSystemCLI()
        try:
            # Initialize system
            cli.agent_manager = AgentManager()
            await cli.agent_manager.initialize()
            
            click.echo("✅ System initialized")
            
            # Test call acquisition workflow
            click.echo("📞 Testing call acquisition workflow...")
            
            # Send a test call to the acquisition agent
            test_call_data = {
                "call_id": "test_call_001",
                "transcript": "Hi, I'd like to schedule a dental appointment for next week. I'm having some pain in my tooth and would like to get it checked out as soon as possible.",
                "timestamp": "2024-01-01T10:00:00Z",
                "caller_info": {"phone": "555-0123"}
            }
            
            await cli.agent_manager.send_message_to_agent(
                "call_acquisition_agent",
                "process_call",
                {"call_data": test_call_data}
            )
            
            # Wait a bit for processing
            await asyncio.sleep(2)
            
            # Check system health
            health = await cli.agent_manager.get_system_health()
            click.echo(f"📊 System health: {health.get('overall_status', 'unknown')}")
            
            # Get system stats
            stats = await cli.agent_manager.get_system_stats()
            click.echo(f"📈 Messages routed: {stats['system_stats'].get('messages_routed', 0)}")
            
            click.echo("✅ Test completed successfully!")
            
        except Exception as e:
            click.echo(f"❌ Test failed: {e}")
            
        finally:
            if cli.agent_manager:
                await cli.agent_manager.shutdown()
    
    asyncio.run(run_test())


if __name__ == "__main__":
    agents()