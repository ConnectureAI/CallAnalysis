"""
Agent Manager for coordinating the AI agent system.

This module provides centralized management and coordination of all
AI agents in the call analysis system, including lifecycle management,
communication routing, and system orchestration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .base_agent import BaseAgent, AgentStatus, AgentMessage, MessageType
from .call_acquisition_agent import CallAcquisitionAgent
from .analysis_orchestrator import AnalysisOrchestrator
from .monitoring_agent import MonitoringAgent
from .workflow_agent import WorkflowAgent
from ..config import get_settings

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Centralized manager for all AI agents.
    
    Handles agent lifecycle, message routing, coordination,
    and provides a unified interface for the agent system.
    """
    
    def __init__(self):
        """Initialize agent manager."""
        self.settings = get_settings()
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        
        # Message routing
        self.message_router = asyncio.Queue()
        self.message_history: List[Dict[str, Any]] = []
        
        # System state
        self.system_status = "initializing"
        self.started_at = None
        self.shutdown_requested = False
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            "agents_registered": 0,
            "agents_running": 0,
            "messages_routed": 0,
            "system_uptime": 0,
            "last_activity": None
        }
    
    async def initialize(self) -> None:
        """Initialize the agent management system."""
        logger.info("Initializing Agent Management System")
        
        try:
            # Load agent configurations
            await self._load_agent_configs()
            
            # Register and initialize agents
            await self._register_agents()
            
            # Start message routing
            self.background_tasks.extend([
                asyncio.create_task(self._message_router_task()),
                asyncio.create_task(self._system_monitor()),
                asyncio.create_task(self._health_checker())
            ])
            
            self.system_status = "running"
            self.started_at = datetime.now()
            
            logger.info("Agent Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Management System: {e}")
            self.system_status = "error"
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the agent management system."""
        logger.info("Shutting down Agent Management System")
        
        self.shutdown_requested = True
        self.system_status = "shutting_down"
        
        try:
            # Stop all agents gracefully
            for agent_id, agent in self.agents.items():
                try:
                    await agent.stop()
                    logger.info(f"Agent {agent_id} stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping agent {agent_id}: {e}")
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Save system state
            await self._save_system_state()
            
            self.system_status = "stopped"
            logger.info("Agent Management System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.system_status = "error"
    
    async def _load_agent_configs(self) -> None:
        """Load agent configurations."""
        # Default agent configurations
        self.agent_configs = {
            "call_acquisition_agent": {
                "enabled": True,
                "auto_start": True,
                "max_concurrent_tasks": 10
            },
            "analysis_orchestrator": {
                "enabled": True,
                "auto_start": True,
                "max_concurrent_tasks": 15
            },
            "monitoring_agent": {
                "enabled": True,
                "auto_start": True,
                "max_concurrent_tasks": 5
            },
            "workflow_agent": {
                "enabled": True,
                "auto_start": True,
                "max_concurrent_tasks": 20
            }
        }
        
        # Try to load from config file
        config_file = self.settings.data_dir / "agent_configs.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_configs = json.load(f)
                    self.agent_configs.update(loaded_configs)
                
                logger.info("Agent configurations loaded from file")
            except Exception as e:
                logger.warning(f"Error loading agent configs: {e}")
    
    async def _register_agents(self) -> None:
        """Register and initialize all agents."""
        # Create agent instances
        agent_classes = {
            "call_acquisition_agent": CallAcquisitionAgent,
            "analysis_orchestrator": AnalysisOrchestrator,
            "monitoring_agent": MonitoringAgent,
            "workflow_agent": WorkflowAgent
        }
        
        for agent_id, agent_class in agent_classes.items():
            config = self.agent_configs.get(agent_id, {})
            
            if not config.get("enabled", False):
                logger.info(f"Agent {agent_id} is disabled, skipping")
                continue
            
            try:
                # Create agent instance
                agent = agent_class()
                
                # Setup message routing for agent
                self._setup_agent_messaging(agent)
                
                # Register agent
                self.agents[agent_id] = agent
                
                # Initialize agent if auto_start is enabled
                if config.get("auto_start", False):
                    await agent.start()
                    logger.info(f"Agent {agent_id} started successfully")
                
                self.stats["agents_registered"] += 1
                
            except Exception as e:
                logger.error(f"Failed to register agent {agent_id}: {e}")
                continue
        
        logger.info(f"Registered {len(self.agents)} agents")
    
    def _setup_agent_messaging(self, agent: BaseAgent) -> None:
        """Setup message routing for an agent."""
        # Replace agent's outbox with our message router
        original_send = agent.send_message
        
        async def routed_send_message(
            recipient: str,
            subject: str,
            payload: Dict[str, Any],
            message_type: MessageType = MessageType.NOTIFICATION,
            priority: int = 2,
            requires_response: bool = False
        ) -> str:
            # Create message
            message = AgentMessage(
                type=message_type,
                sender=agent.agent_id,
                recipient=recipient,
                subject=subject,
                payload=payload,
                priority=priority,
                requires_response=requires_response
            )
            
            # Route through our system
            await self.message_router.put(message)
            self.stats["messages_routed"] += 1
            
            return message.id
        
        # Replace the agent's send_message method
        agent.send_message = routed_send_message
    
    async def _message_router_task(self) -> None:
        """Route messages between agents."""
        while not self.shutdown_requested:
            try:
                # Get message from router queue
                message = await asyncio.wait_for(self.message_router.get(), timeout=1.0)
                
                # Route message to recipient
                await self._route_message(message)
                
                # Store in message history
                self.message_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "from": message.sender,
                    "to": message.recipient,
                    "subject": message.subject,
                    "type": message.type.value,
                    "priority": message.priority
                })
                
                # Keep only recent message history
                if len(self.message_history) > 1000:
                    self.message_history = self.message_history[-500:]
                
                self.stats["last_activity"] = datetime.now().isoformat()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message router: {e}")
    
    async def _route_message(self, message: AgentMessage) -> None:
        """Route a message to its recipient."""
        recipient_id = message.recipient
        
        if recipient_id in self.agents:
            recipient_agent = self.agents[recipient_id]
            
            try:
                # Put message in recipient's inbox
                await recipient_agent.inbox.put(message)
                
                logger.debug(f"Routed message from {message.sender} to {recipient_id}: {message.subject}")
                
            except Exception as e:
                logger.error(f"Error routing message to {recipient_id}: {e}")
        
        elif recipient_id == "broadcast":
            # Broadcast to all agents except sender
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender:
                    try:
                        await agent.inbox.put(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting message to {agent_id}: {e}")
        
        else:
            logger.warning(f"Unknown message recipient: {recipient_id}")
    
    async def _system_monitor(self) -> None:
        """Monitor overall system health."""
        while not self.shutdown_requested:
            try:
                # Update system statistics
                running_agents = sum(
                    1 for agent in self.agents.values()
                    if agent.status in [AgentStatus.IDLE, AgentStatus.RUNNING, AgentStatus.BUSY]
                )
                
                self.stats["agents_running"] = running_agents
                
                if self.started_at:
                    self.stats["system_uptime"] = (datetime.now() - self.started_at).total_seconds()
                
                # Check for agent failures
                for agent_id, agent in self.agents.items():
                    if agent.status == AgentStatus.ERROR:
                        logger.warning(f"Agent {agent_id} is in error state")
                        
                        # Attempt restart if configured
                        config = self.agent_configs.get(agent_id, {})
                        if config.get("auto_restart", False):
                            try:
                                await agent.start()
                                logger.info(f"Successfully restarted agent {agent_id}")
                            except Exception as e:
                                logger.error(f"Failed to restart agent {agent_id}: {e}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(60)
    
    async def _health_checker(self) -> None:
        """Perform periodic system health checks."""
        while not self.shutdown_requested:
            try:
                health_report = await self.get_system_health()
                
                # Log health status
                overall_status = health_report.get("overall_status", "unknown")
                if overall_status != "healthy":
                    logger.warning(f"System health check: {overall_status}")
                    
                    # Log specific issues
                    issues = health_report.get("issues", [])
                    for issue in issues:
                        logger.warning(f"Health issue: {issue}")
                
                await asyncio.sleep(300)  # Health check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(60)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        health_report = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "status": self.system_status,
                "uptime": self.stats["system_uptime"],
                "agents_registered": len(self.agents),
                "agents_running": self.stats["agents_running"]
            },
            "agents": {},
            "issues": []
        }
        
        try:
            # Check each agent's health
            for agent_id, agent in self.agents.items():
                agent_status = agent.get_status()
                health_report["agents"][agent_id] = agent_status
                
                # Check for issues
                if agent_status["status"] == "error":
                    health_report["issues"].append(f"Agent {agent_id} is in error state")
                elif agent_status["status"] == "stopped":
                    health_report["issues"].append(f"Agent {agent_id} is not running")
                elif agent_status.get("queue_size", 0) > 50:
                    health_report["issues"].append(f"Agent {agent_id} has large queue: {agent_status['queue_size']}")
            
            # Check message routing health
            if len(self.message_history) == 0 and self.started_at:
                uptime_minutes = (datetime.now() - self.started_at).seconds / 60
                if uptime_minutes > 5:  # No messages for 5+ minutes
                    health_report["issues"].append("No message activity detected")
            
            # Determine overall status
            if health_report["issues"]:
                if len(health_report["issues"]) >= 3:
                    health_report["overall_status"] = "critical"
                else:
                    health_report["overall_status"] = "degraded"
        
        except Exception as e:
            health_report["error"] = str(e)
            health_report["overall_status"] = "error"
        
        return health_report
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a specific agent."""
        if agent_id in self.agents:
            return self.agents[agent_id].get_status()
        else:
            return {"error": f"Agent {agent_id} not found"}
    
    async def send_message_to_agent(
        self,
        agent_id: str,
        subject: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.COMMAND
    ) -> str:
        """Send a message to a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        message = AgentMessage(
            type=message_type,
            sender="agent_manager",
            recipient=agent_id,
            subject=subject,
            payload=payload
        )
        
        await self.message_router.put(message)
        return message.id
    
    async def start_agent(self, agent_id: str) -> Dict[str, Any]:
        """Start a specific agent."""
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}
        
        try:
            agent = self.agents[agent_id]
            if agent.status == AgentStatus.STOPPED:
                await agent.start()
                return {"status": "started", "agent_id": agent_id}
            else:
                return {"status": "already_running", "agent_id": agent_id}
        
        except Exception as e:
            return {"error": str(e), "agent_id": agent_id}
    
    async def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        """Stop a specific agent."""
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}
        
        try:
            agent = self.agents[agent_id]
            if agent.status != AgentStatus.STOPPED:
                await agent.stop()
                return {"status": "stopped", "agent_id": agent_id}
            else:
                return {"status": "already_stopped", "agent_id": agent_id}
        
        except Exception as e:
            return {"error": str(e), "agent_id": agent_id}
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "system_stats": self.stats,
            "message_history_size": len(self.message_history),
            "recent_messages": self.message_history[-10:] if self.message_history else [],
            "agent_summary": {
                agent_id: {
                    "status": agent.status.value,
                    "active_tasks": len(agent.active_tasks),
                    "queue_size": agent.task_queue.qsize()
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    async def _save_system_state(self) -> None:
        """Save system state to disk."""
        try:
            state_file = self.settings.data_dir / "agent_system_state.json"
            
            state_data = {
                "shutdown_time": datetime.now().isoformat(),
                "system_stats": self.stats,
                "agent_configs": self.agent_configs,
                "message_history_summary": {
                    "total_messages": len(self.message_history),
                    "last_messages": self.message_history[-10:] if self.message_history else []
                }
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info("Agent system state saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent IDs."""
        return list(self.agents.keys())
    
    def is_agent_running(self, agent_id: str) -> bool:
        """Check if an agent is running."""
        if agent_id in self.agents:
            return self.agents[agent_id].status in [
                AgentStatus.IDLE, AgentStatus.RUNNING, AgentStatus.BUSY
            ]
        return False