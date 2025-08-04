"""
Monitoring Agent for system health and performance tracking.

This agent monitors the entire call analysis system, tracks performance
metrics, manages alerts, and provides system health insights.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import psutil
from pathlib import Path

from .base_agent import BaseAgent, AgentStatus, AgentCapability, MessageType
from ..config import get_settings

logger = logging.getLogger(__name__)


class SystemAlert:
    """Represents a system alert."""
    
    def __init__(
        self,
        alert_id: str,
        severity: str,
        message: str,
        component: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.alert_id = alert_id
        self.severity = severity  # low, medium, high, critical
        self.message = message
        self.component = component
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.acknowledged = False
        self.resolved = False


class MonitoringAgent(BaseAgent):
    """
    Monitors system health, performance, and generates alerts.
    
    Tracks agent status, resource usage, analysis performance,
    and system reliability metrics.
    """
    
    def __init__(
        self,
        agent_id: str = "monitoring_agent",
        name: str = "System Monitoring Agent"
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="Monitors system health and performance",
            max_concurrent_tasks=5
        )
        
        self.settings = get_settings()
        
        # Monitoring data
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.alerts: Dict[str, SystemAlert] = {}
        
        # Alert thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "agent_response_time": 30.0,
            "analysis_failure_rate": 0.15,
            "queue_backup_size": 100
        }
        
        # Statistics
        self.stats = {
            "active_agents": 0,
            "total_alerts": 0,
            "critical_alerts": 0,
            "system_uptime": 0,
            "last_health_check": None,
            "average_response_time": 0.0
        }
        
        # Setup capabilities
        self._setup_capabilities()
        
        # Register message handlers
        self.register_handler("heartbeat", self._handle_agent_heartbeat)
        self.register_handler("analysis_completed", self._handle_analysis_completed)
        self.register_handler("agent_status", self._handle_agent_status)
        self.register_handler("get_system_health", self._handle_get_system_health)
        self.register_handler("get_alerts", self._handle_get_alerts)
        self.register_handler("acknowledge_alert", self._handle_acknowledge_alert)
    
    def _setup_capabilities(self):
        """Setup monitoring capabilities."""
        capabilities = [
            AgentCapability(
                name="system_monitoring",
                description="Monitor system resources and performance",
                input_types=["system_metrics"],
                output_types=["health_status", "alerts"],
                estimated_duration=timedelta(seconds=10)
            ),
            AgentCapability(
                name="agent_monitoring",
                description="Monitor agent health and performance",
                input_types=["agent_status", "heartbeat_data"],
                output_types=["agent_health", "availability_metrics"],
                estimated_duration=timedelta(seconds=5)
            ),
            AgentCapability(
                name="performance_tracking",
                description="Track and analyze system performance trends",
                input_types=["performance_data"],
                output_types=["performance_reports", "trend_analysis"],
                estimated_duration=timedelta(seconds=15)
            ),
            AgentCapability(
                name="alert_management",
                description="Generate and manage system alerts",
                input_types=["monitoring_data", "thresholds"],
                output_types=["alerts", "notifications"],
                estimated_duration=timedelta(seconds=5)
            )
        ]
        
        for capability in capabilities:
            self.add_capability(capability)
    
    async def initialize(self) -> None:
        """Initialize the monitoring agent."""
        logger.info("Initializing System Monitoring Agent")
        
        # Load existing monitoring data
        await self._load_monitoring_data()
        
        # Start monitoring tasks
        self._background_tasks.extend([
            asyncio.create_task(self._system_monitor()),
            asyncio.create_task(self._agent_monitor()),
            asyncio.create_task(self._performance_tracker()),
            asyncio.create_task(self._alert_manager()),
            asyncio.create_task(self._health_checker())
        ])
        
        logger.info("System Monitoring Agent initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup monitoring agent resources."""
        logger.info("Cleaning up System Monitoring Agent")
        
        # Save monitoring data
        await self._save_monitoring_data()
        
        # Generate final system report
        await self._generate_shutdown_report()
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process monitoring tasks."""
        task_type = task.get("type")
        task_data = task.get("data", {})
        
        if task_type == "system_health_check":
            return await self._perform_system_health_check()
        elif task_type == "generate_report":
            return await self._generate_performance_report(task_data)
        elif task_type == "check_alerts":
            return await self._check_alert_conditions()
        elif task_type == "agent_health_check":
            return await self._check_agent_health(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _system_monitor(self) -> None:
        """Monitor system resources."""
        while self._running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                system_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "memory_available": memory.available / (1024**3),  # GB
                    "disk_usage": disk.percent,
                    "disk_free": disk.free / (1024**3),  # GB
                    "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                }
                
                self.system_metrics = system_metrics
                
                # Check for threshold violations
                await self._check_system_thresholds(system_metrics)
                
                # Store in performance history
                self.performance_history.append(system_metrics)
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.performance_history = [
                    entry for entry in self.performance_history
                    if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
                ]
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(60)
    
    async def _agent_monitor(self) -> None:
        """Monitor agent health and status."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Check for stale agents
                for agent_id, status_data in list(self.agent_status.items()):
                    last_heartbeat = status_data.get("last_heartbeat")
                    if last_heartbeat:
                        last_beat = datetime.fromisoformat(last_heartbeat)
                        if (current_time - last_beat).seconds > 120:  # No heartbeat for 2 minutes
                            await self._create_alert(
                                "high",
                                f"Agent {agent_id} has not sent heartbeat for over 2 minutes",
                                agent_id
                            )
                
                # Update statistics
                self.stats["active_agents"] = len([
                    agent for agent, data in self.agent_status.items()
                    if data.get("status") in ["idle", "running", "busy"]
                ])
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in agent monitor: {e}")
                await asyncio.sleep(30)
    
    async def _performance_tracker(self) -> None:
        """Track performance trends and patterns."""
        while self._running:
            try:
                if len(self.performance_history) >= 10:
                    # Calculate averages for last 10 measurements
                    recent_metrics = self.performance_history[-10:]
                    
                    avg_cpu = sum(m["cpu_usage"] for m in recent_metrics) / len(recent_metrics)
                    avg_memory = sum(m["memory_usage"] for m in recent_metrics) / len(recent_metrics)
                    
                    # Detect performance degradation trends
                    if avg_cpu > self.thresholds["cpu_usage"]:
                        await self._create_alert(
                            "medium",
                            f"High CPU usage detected: {avg_cpu:.1f}%",
                            "system"
                        )
                    
                    if avg_memory > self.thresholds["memory_usage"]:
                        await self._create_alert(
                            "medium",
                            f"High memory usage detected: {avg_memory:.1f}%",
                            "system"
                        )
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracker: {e}")
                await asyncio.sleep(60)
    
    async def _alert_manager(self) -> None:
        """Manage system alerts."""
        while self._running:
            try:
                # Clean up old resolved alerts
                cutoff_time = datetime.now() - timedelta(hours=48)
                alerts_to_remove = [
                    alert_id for alert_id, alert in self.alerts.items()
                    if alert.resolved and alert.timestamp < cutoff_time
                ]
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                
                # Update alert statistics
                self.stats["total_alerts"] = len(self.alerts)
                self.stats["critical_alerts"] = len([
                    alert for alert in self.alerts.values()
                    if alert.severity == "critical" and not alert.resolved
                ])
                
                await asyncio.sleep(600)  # Clean up every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in alert manager: {e}")
                await asyncio.sleep(60)
    
    async def _health_checker(self) -> None:
        """Perform periodic comprehensive health checks."""
        while self._running:
            try:
                health_report = await self._perform_system_health_check()
                self.stats["last_health_check"] = datetime.now().isoformat()
                
                # Log health status
                overall_status = health_report.get("overall_status", "unknown")
                logger.info(f"System health check completed: {overall_status}")
                
                await asyncio.sleep(1800)  # Health check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(300)
    
    async def _check_system_thresholds(self, metrics: Dict[str, Any]) -> None:
        """Check if system metrics exceed thresholds."""
        try:
            if metrics["cpu_usage"] > self.thresholds["cpu_usage"]:
                await self._create_alert(
                    "high",
                    f"CPU usage critical: {metrics['cpu_usage']:.1f}%",
                    "system",
                    {"cpu_usage": metrics["cpu_usage"]}
                )
            
            if metrics["memory_usage"] > self.thresholds["memory_usage"]:
                await self._create_alert(
                    "high",
                    f"Memory usage critical: {metrics['memory_usage']:.1f}%",
                    "system",
                    {"memory_usage": metrics["memory_usage"]}
                )
            
            if metrics["disk_usage"] > self.thresholds["disk_usage"]:
                await self._create_alert(
                    "critical",
                    f"Disk usage critical: {metrics['disk_usage']:.1f}%",
                    "system",
                    {"disk_usage": metrics["disk_usage"]}
                )
        
        except Exception as e:
            logger.error(f"Error checking system thresholds: {e}")
    
    async def _create_alert(
        self,
        severity: str,
        message: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new system alert."""
        alert_id = f"alert_{datetime.now().timestamp()}"
        
        alert = SystemAlert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            component=component,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Check if similar alert already exists (avoid spam)
        similar_alerts = [
            existing_alert for existing_alert in self.alerts.values()
            if (existing_alert.component == component and 
                existing_alert.message == message and
                not existing_alert.resolved and
                (datetime.now() - existing_alert.timestamp).seconds < 3600)  # Within last hour
        ]
        
        if not similar_alerts:
            self.alerts[alert_id] = alert
            logger.warning(f"Alert created [{severity}]: {message} (Component: {component})")
            
            # Send notification for critical alerts
            if severity == "critical":
                await self._send_critical_alert_notification(alert)
        
        return alert_id
    
    async def _send_critical_alert_notification(self, alert: SystemAlert) -> None:
        """Send notification for critical alerts."""
        try:
            # Broadcast to all agents
            await self.broadcast_message(
                "critical_alert",
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "message": alert.message,
                    "component": alert.component,
                    "timestamp": alert.timestamp.isoformat()
                },
                message_type=MessageType.NOTIFICATION,
                priority=4
            )
            
        except Exception as e:
            logger.error(f"Error sending critical alert notification: {e}")
    
    async def _perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_report = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check system resources
            if self.system_metrics:
                cpu_status = "healthy"
                memory_status = "healthy"
                disk_status = "healthy"
                
                if self.system_metrics["cpu_usage"] > 70:
                    cpu_status = "warning"
                    health_report["issues"].append("High CPU usage")
                if self.system_metrics["cpu_usage"] > 90:
                    cpu_status = "critical"
                
                if self.system_metrics["memory_usage"] > 80:
                    memory_status = "warning"
                    health_report["issues"].append("High memory usage")
                if self.system_metrics["memory_usage"] > 95:
                    memory_status = "critical"
                
                if self.system_metrics["disk_usage"] > 85:
                    disk_status = "warning"
                    health_report["issues"].append("High disk usage")
                if self.system_metrics["disk_usage"] > 95:
                    disk_status = "critical"
                
                health_report["components"]["system"] = {
                    "cpu": cpu_status,
                    "memory": memory_status,
                    "disk": disk_status,
                    "metrics": self.system_metrics
                }
            
            # Check agent health
            agent_health = {}
            for agent_id, status_data in self.agent_status.items():
                agent_status = status_data.get("status", "unknown")
                last_heartbeat = status_data.get("last_heartbeat")
                
                if last_heartbeat:
                    last_beat = datetime.fromisoformat(last_heartbeat)
                    minutes_since = (datetime.now() - last_beat).seconds / 60
                    
                    if minutes_since > 5:
                        agent_health[agent_id] = "stale"
                        health_report["issues"].append(f"Agent {agent_id} is not responding")
                    elif agent_status == "error":
                        agent_health[agent_id] = "error"
                        health_report["issues"].append(f"Agent {agent_id} is in error state")
                    else:
                        agent_health[agent_id] = "healthy"
                else:
                    agent_health[agent_id] = "unknown"
            
            health_report["components"]["agents"] = agent_health
            
            # Check alerts
            active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
            critical_alerts = [alert for alert in active_alerts if alert.severity == "critical"]
            
            if critical_alerts:
                health_report["issues"].append(f"{len(critical_alerts)} critical alerts active")
                health_report["components"]["alerts"] = "critical"
            elif active_alerts:
                health_report["components"]["alerts"] = "warning"
            else:
                health_report["components"]["alerts"] = "healthy"
            
            # Determine overall status
            if health_report["issues"]:
                if any("critical" in issue.lower() for issue in health_report["issues"]):
                    health_report["overall_status"] = "critical"
                else:
                    health_report["overall_status"] = "degraded"
            
            # Generate recommendations
            if health_report["issues"]:
                health_report["recommendations"] = self._generate_health_recommendations(health_report)
        
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            health_report["error"] = str(e)
            health_report["overall_status"] = "error"
        
        return health_report
    
    def _generate_health_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        for issue in health_report["issues"]:
            if "high cpu" in issue.lower():
                recommendations.append("Consider scaling up compute resources or optimizing CPU-intensive processes")
            elif "high memory" in issue.lower():
                recommendations.append("Monitor memory leaks and consider increasing available RAM")
            elif "high disk" in issue.lower():
                recommendations.append("Clean up old files and consider expanding disk storage")
            elif "not responding" in issue.lower():
                recommendations.append("Investigate and restart unresponsive agents")
            elif "critical alert" in issue.lower():
                recommendations.append("Address critical alerts immediately to prevent system failure")
        
        return recommendations
    
    async def _load_monitoring_data(self) -> None:
        """Load existing monitoring data."""
        try:
            data_file = self.settings.data_dir / "monitoring_data.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    
                self.agent_status = data.get("agent_status", {})
                self.stats = data.get("stats", self.stats)
                
                logger.info("Monitoring data loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
    
    async def _save_monitoring_data(self) -> None:
        """Save monitoring data to disk."""
        try:
            data_file = self.settings.data_dir / "monitoring_data.json"
            
            data = {
                "agent_status": self.agent_status,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info("Monitoring data saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    async def _generate_shutdown_report(self) -> None:
        """Generate system report on shutdown."""
        try:
            report = {
                "shutdown_time": datetime.now().isoformat(),
                "uptime": (datetime.now() - self.created_at).total_seconds(),
                "final_stats": self.stats,
                "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
                "total_agents_monitored": len(self.agent_status)
            }
            
            report_file = self.settings.data_dir / f"shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Shutdown report saved: {report_file}")
        
        except Exception as e:
            logger.error(f"Error generating shutdown report: {e}")
    
    # Message handlers
    
    async def _handle_agent_heartbeat(self, message) -> None:
        """Handle agent heartbeat messages."""
        sender = message.sender
        payload = message.payload
        
        self.agent_status[sender] = {
            "status": payload.get("status", "unknown"),
            "last_heartbeat": datetime.now().isoformat(),
            "timestamp": payload.get("timestamp"),
            "metrics": payload.get("metrics", {})
        }
        
        # Send heartbeat acknowledgment
        await self.send_message(
            sender,
            "heartbeat_ack",
            {"timestamp": datetime.now().isoformat()},
            MessageType.RESPONSE,
            priority=1
        )
    
    async def _handle_analysis_completed(self, message) -> None:
        """Handle analysis completion notifications."""
        payload = message.payload
        call_id = payload.get("call_id")
        duration = payload.get("duration", 0)
        
        # Track analysis performance
        if duration > self.thresholds["agent_response_time"]:
            await self._create_alert(
                "medium",
                f"Slow analysis detected for call {call_id}: {duration:.2f}s",
                "analysis_orchestrator",
                {"call_id": call_id, "duration": duration}
            )
    
    async def _handle_agent_status(self, message) -> None:
        """Handle agent status updates."""
        sender = message.sender
        payload = message.payload
        
        if sender not in self.agent_status:
            self.agent_status[sender] = {}
        
        self.agent_status[sender].update({
            "status": payload.get("status"),
            "last_update": datetime.now().isoformat(),
            "details": payload
        })
    
    async def _handle_get_system_health(self, message) -> None:
        """Handle system health requests."""
        health_report = await self._perform_system_health_check()
        
        await self.send_message(
            message.sender,
            "system_health_response",
            health_report,
            MessageType.RESPONSE
        )
    
    async def _handle_get_alerts(self, message) -> None:
        """Handle alert requests."""
        severity_filter = message.payload.get("severity")
        resolved_filter = message.payload.get("resolved", False)
        
        filtered_alerts = []
        for alert in self.alerts.values():
            if severity_filter and alert.severity != severity_filter:
                continue
            if not resolved_filter and alert.resolved:
                continue
            
            filtered_alerts.append({
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "message": alert.message,
                "component": alert.component,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            })
        
        await self.send_message(
            message.sender,
            "alerts_response",
            {"alerts": filtered_alerts},
            MessageType.RESPONSE
        )
    
    async def _handle_acknowledge_alert(self, message) -> None:
        """Handle alert acknowledgment."""
        alert_id = message.payload.get("alert_id")
        
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            response = {"status": "acknowledged", "alert_id": alert_id}
        else:
            response = {"error": f"Alert {alert_id} not found"}
        
        await self.send_message(
            message.sender,
            "alert_ack_response",
            response,
            MessageType.RESPONSE
        )