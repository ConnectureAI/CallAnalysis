"""
Base agent class for the AI agent system.

This module provides the foundation for all AI agents in the system,
including communication protocols, state management, and coordination.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import uuid
import json

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status enumeration."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


class MessageType(Enum):
    """Message types for agent communication."""
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.NOTIFICATION
    sender: str = ""
    recipient: str = ""
    subject: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical
    requires_response: bool = False
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "subject": self.subject,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "notification")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            subject=data.get("subject", ""),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            priority=data.get("priority", 1),
            requires_response=data.get("requires_response", False),
            correlation_id=data.get("correlation_id")
        )


@dataclass
class AgentCapability:
    """Describes an agent's capability."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    requires_approval: bool = False
    estimated_duration: Optional[timedelta] = None


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration: float = 0.0
    last_activity: Optional[datetime] = None
    uptime: timedelta = timedelta()
    error_count: int = 0
    messages_sent: int = 0
    messages_received: int = 0


class BaseAgent(ABC):
    """
    Base class for all AI agents in the system.
    
    Provides common functionality for communication, state management,
    task execution, and coordination with other agents.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        max_concurrent_tasks: int = 5
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable agent name
            description: Agent description
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # State management
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.now()
        self.last_heartbeat = datetime.now()
        
        # Task management
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks: List[Dict[str, Any]] = []
        
        # Communication
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        self.message_handlers: Dict[str, Callable] = {}
        self.subscriptions: List[str] = []
        
        # Capabilities and metrics
        self.capabilities: List[AgentCapability] = []
        self.metrics = AgentMetrics()
        
        # Coordination
        self.parent_agent: Optional[str] = None
        self.child_agents: List[str] = []
        self.peer_agents: List[str] = []
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # Event loop and tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Setup default message handlers
        self._setup_default_handlers()
        
        logger.info(f"Agent {self.name} ({self.agent_id}) initialized")
    
    def _setup_default_handlers(self):
        """Setup default message handlers."""
        self.register_handler("heartbeat", self._handle_heartbeat)
        self.register_handler("status_query", self._handle_status_query)
        self.register_handler("capabilities_query", self._handle_capabilities_query)
        self.register_handler("stop", self._handle_stop)
        self.register_handler("pause", self._handle_pause)
        self.register_handler("resume", self._handle_resume)
    
    async def start(self) -> None:
        """Start the agent."""
        logger.info(f"Starting agent {self.name}")
        
        self._running = True
        self.status = AgentStatus.IDLE
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._task_processor()),
            asyncio.create_task(self._heartbeat_sender()),
            asyncio.create_task(self._health_monitor())
        ]
        
        # Agent-specific initialization
        try:
            await self.initialize()
            logger.info(f"Agent {self.name} started successfully")
        except Exception as e:
            logger.error(f"Failed to start agent {self.name}: {e}")
            self.status = AgentStatus.ERROR
            raise
    
    async def stop(self) -> None:
        """Stop the agent."""
        logger.info(f"Stopping agent {self.name}")
        
        self._running = False
        self.status = AgentStatus.STOPPED
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Agent-specific cleanup
        await self.cleanup()
        
        logger.info(f"Agent {self.name} stopped")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent-specific resources."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup agent-specific resources."""
        pass
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single task.
        
        Args:
            task: Task dictionary containing task details
            
        Returns:
            Task result dictionary
        """
        pass
    
    async def send_message(
        self, 
        recipient: str, 
        subject: str, 
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.NOTIFICATION,
        priority: int = 2,
        requires_response: bool = False
    ) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient: Recipient agent ID
            subject: Message subject
            payload: Message payload
            message_type: Type of message
            priority: Message priority (1-4)
            requires_response: Whether response is required
            
        Returns:
            Message ID
        """
        message = AgentMessage(
            type=message_type,
            sender=self.agent_id,
            recipient=recipient,
            subject=subject,
            payload=payload,
            priority=priority,
            requires_response=requires_response
        )
        
        await self.outbox.put(message)
        self.metrics.messages_sent += 1
        
        logger.debug(f"Agent {self.name} sent message to {recipient}: {subject}")
        return message.id
    
    async def broadcast_message(
        self,
        subject: str,
        payload: Dict[str, Any],
        recipients: Optional[List[str]] = None,
        message_type: MessageType = MessageType.NOTIFICATION,
        priority: int = 2
    ) -> List[str]:
        """
        Broadcast message to multiple recipients.
        
        Args:
            subject: Message subject
            payload: Message payload
            recipients: List of recipient IDs (None for all peers)
            message_type: Type of message
            priority: Message priority
            
        Returns:
            List of message IDs
        """
        if recipients is None:
            recipients = self.peer_agents + self.child_agents
        
        message_ids = []
        for recipient in recipients:
            message_id = await self.send_message(
                recipient, subject, payload, message_type, priority
            )
            message_ids.append(message_id)
        
        return message_ids
    
    async def submit_task(
        self, 
        task_type: str, 
        task_data: Dict[str, Any],
        priority: int = 2
    ) -> str:
        """
        Submit a task for processing.
        
        Args:
            task_type: Type of task
            task_data: Task data
            priority: Task priority
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "type": task_type,
            "data": task_data,
            "priority": priority,
            "created_at": datetime.now(),
            "status": "queued",
            "agent_id": self.agent_id
        }
        
        await self.task_queue.put(task)
        logger.debug(f"Agent {self.name} submitted task {task_id}: {task_type}")
        
        return task_id
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Agent {self.name} registered handler for {message_type}")
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent."""
        self.capabilities.append(capability)
        logger.debug(f"Agent {self.name} added capability: {capability.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "uptime": (datetime.now() - self.created_at).total_seconds(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "messages_sent": self.metrics.messages_sent,
                "messages_received": self.metrics.messages_received,
                "error_count": self.metrics.error_count
            }
        }
    
    async def _message_processor(self) -> None:
        """Process incoming messages."""
        while self._running:
            try:
                # Get message from inbox
                message = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                self.metrics.messages_received += 1
                self.metrics.last_activity = datetime.now()
                
                # Find and execute handler
                handler = self.message_handlers.get(message.subject)
                if handler:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error handling message {message.subject}: {e}")
                        await self._send_error_response(message, str(e))
                else:
                    logger.warning(f"No handler for message type: {message.subject}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                self.metrics.error_count += 1
    
    async def _task_processor(self) -> None:
        """Process tasks from the queue."""
        while self._running:
            try:
                # Check if we can handle more tasks
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Start task processing
                task_id = task["id"]
                self.active_tasks[task_id] = task
                task["status"] = "running"
                task["started_at"] = datetime.now()
                
                # Update status
                if self.status == AgentStatus.IDLE:
                    self.status = AgentStatus.RUNNING
                
                # Process task in background
                asyncio.create_task(self._execute_task(task))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                self.metrics.error_count += 1
    
    async def _execute_task(self, task: Dict[str, Any]) -> None:
        """Execute a single task."""
        task_id = task["id"]
        start_time = datetime.now()
        
        try:
            # Process the task
            result = await self.process_task(task)
            
            # Update task status
            task["status"] = "completed"
            task["completed_at"] = datetime.now()
            task["result"] = result
            task["duration"] = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self._update_average_duration(task["duration"])
            
            logger.debug(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            task["status"] = "failed"
            task["error"] = str(e)
            task["completed_at"] = datetime.now()
            
            self.metrics.tasks_failed += 1
            logger.error(f"Task {task_id} failed: {e}")
        
        finally:
            # Move task to completed list and remove from active
            self.completed_tasks.append(task)
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Update status if no active tasks
            if not self.active_tasks and self.task_queue.empty():
                self.status = AgentStatus.IDLE
    
    async def _heartbeat_sender(self) -> None:
        """Send periodic heartbeat messages."""
        while self._running:
            try:
                self.last_heartbeat = datetime.now()
                
                # Send heartbeat to parent if exists
                if self.parent_agent:
                    await self.send_message(
                        self.parent_agent,
                        "heartbeat",
                        {"status": self.status.value, "timestamp": self.last_heartbeat.isoformat()},
                        MessageType.HEARTBEAT,
                        priority=1
                    )
                
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitor(self) -> None:
        """Monitor agent health."""
        while self._running:
            try:
                # Check for stuck tasks
                current_time = datetime.now()
                for task_id, task in list(self.active_tasks.items()):
                    if "started_at" in task:
                        duration = current_time - task["started_at"]
                        if duration > timedelta(minutes=30):  # Task running too long
                            logger.warning(f"Task {task_id} has been running for {duration}")
                
                # Update uptime
                self.metrics.uptime = current_time - self.created_at
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)
    
    def _update_average_duration(self, duration: float) -> None:
        """Update average task duration."""
        total_tasks = self.metrics.tasks_completed
        if total_tasks == 1:
            self.metrics.average_task_duration = duration
        else:
            # Running average
            current_avg = self.metrics.average_task_duration
            self.metrics.average_task_duration = (
                (current_avg * (total_tasks - 1) + duration) / total_tasks
            )
    
    async def _send_error_response(self, original_message: AgentMessage, error: str) -> None:
        """Send error response to message sender."""
        if original_message.requires_response:
            await self.send_message(
                original_message.sender,
                "error_response",
                {
                    "original_message_id": original_message.id,
                    "error": error
                },
                MessageType.ERROR,
                priority=3,
                requires_response=False
            )
    
    # Default message handlers
    
    async def _handle_heartbeat(self, message: AgentMessage) -> None:
        """Handle heartbeat message."""
        # Acknowledge heartbeat
        if message.requires_response:
            await self.send_message(
                message.sender,
                "heartbeat_ack",
                {"timestamp": datetime.now().isoformat()},
                MessageType.RESPONSE
            )
    
    async def _handle_status_query(self, message: AgentMessage) -> None:
        """Handle status query message."""
        status = self.get_status()
        await self.send_message(
            message.sender,
            "status_response",
            status,
            MessageType.RESPONSE
        )
    
    async def _handle_capabilities_query(self, message: AgentMessage) -> None:
        """Handle capabilities query message."""
        capabilities = [
            {
                "name": cap.name,
                "description": cap.description,
                "input_types": cap.input_types,
                "output_types": cap.output_types,
                "requires_approval": cap.requires_approval
            }
            for cap in self.capabilities
        ]
        
        await self.send_message(
            message.sender,
            "capabilities_response",
            {"capabilities": capabilities},
            MessageType.RESPONSE
        )
    
    async def _handle_stop(self, message: AgentMessage) -> None:
        """Handle stop command."""
        logger.info(f"Agent {self.name} received stop command")
        await self.stop()
    
    async def _handle_pause(self, message: AgentMessage) -> None:
        """Handle pause command."""
        if self.status in [AgentStatus.IDLE, AgentStatus.RUNNING]:
            self.status = AgentStatus.MAINTENANCE
            logger.info(f"Agent {self.name} paused")
    
    async def _handle_resume(self, message: AgentMessage) -> None:
        """Handle resume command."""
        if self.status == AgentStatus.MAINTENANCE:
            self.status = AgentStatus.IDLE
            logger.info(f"Agent {self.name} resumed")