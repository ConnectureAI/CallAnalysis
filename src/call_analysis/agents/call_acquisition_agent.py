"""
Call Acquisition Agent for automatic call monitoring and retrieval.

This agent automatically monitors various call sources, acquires new calls,
and processes them for analysis. It can integrate with multiple systems
including phone systems, call recording platforms, and file systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import aiofiles
import aiohttp
from ftplib import FTP
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from .base_agent import BaseAgent, AgentStatus, AgentCapability, MessageType
from ..config import get_settings
from ..models import CallInsight

logger = logging.getLogger(__name__)


class CallSource:
    """Represents a call source configuration."""
    
    def __init__(
        self,
        source_id: str,
        source_type: str,
        name: str,
        config: Dict[str, Any],
        enabled: bool = True,
        polling_interval: int = 300  # 5 minutes default
    ):
        self.source_id = source_id
        self.source_type = source_type
        self.name = name
        self.config = config
        self.enabled = enabled
        self.polling_interval = polling_interval
        self.last_check = None
        self.total_calls_acquired = 0
        self.errors = []


class CallAcquisitionAgent(BaseAgent):
    """
    Agent responsible for acquiring new calls from various sources.
    
    Supports multiple call sources:
    - File system monitoring
    - FTP/SFTP servers
    - HTTP/REST APIs
    - Email attachments
    - Database polling
    - Webhook receivers
    """
    
    def __init__(
        self,
        agent_id: str = "call_acquisition_agent",
        name: str = "Call Acquisition Agent"
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="Automatically acquires and processes new calls from various sources",
            max_concurrent_tasks=10
        )
        
        self.settings = get_settings()
        self.call_sources: Dict[str, CallSource] = {}
        self.acquired_calls: List[Dict[str, Any]] = []
        self.processing_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            "total_calls_acquired": 0,
            "successful_acquisitions": 0,
            "failed_acquisitions": 0,
            "last_acquisition": None,
            "sources_monitored": 0
        }
        
        # Add capabilities
        self._setup_capabilities()
        
        # Setup message handlers
        self.register_handler("add_source", self._handle_add_source)
        self.register_handler("remove_source", self._handle_remove_source)
        self.register_handler("list_sources", self._handle_list_sources)
        self.register_handler("get_stats", self._handle_get_stats)
        self.register_handler("manual_check", self._handle_manual_check)
    
    def _setup_capabilities(self):
        """Setup agent capabilities."""
        capabilities = [
            AgentCapability(
                name="file_system_monitoring",
                description="Monitor file system directories for new call files",
                input_types=["directory_path", "file_patterns"],
                output_types=["call_data"],
                estimated_duration=timedelta(seconds=30)
            ),
            AgentCapability(
                name="ftp_monitoring",
                description="Monitor FTP/SFTP servers for new call recordings",
                input_types=["ftp_config"],
                output_types=["call_data"],
                estimated_duration=timedelta(minutes=2)
            ),
            AgentCapability(
                name="api_polling",
                description="Poll REST APIs for new call data",
                input_types=["api_config"],
                output_types=["call_data"],
                estimated_duration=timedelta(minutes=1)
            ),
            AgentCapability(
                name="email_monitoring",
                description="Monitor email for call attachments",
                input_types=["email_config"],
                output_types=["call_data"],
                estimated_duration=timedelta(minutes=3)
            ),
            AgentCapability(
                name="webhook_receiver",
                description="Receive webhook notifications for new calls",
                input_types=["webhook_data"],
                output_types=["call_data"],
                estimated_duration=timedelta(seconds=5)
            )
        ]
        
        for capability in capabilities:
            self.add_capability(capability)
    
    async def initialize(self) -> None:
        """Initialize the call acquisition agent."""
        logger.info("Initializing Call Acquisition Agent")
        
        # Load existing call sources from configuration
        await self._load_call_sources()
        
        # Start monitoring tasks
        self._background_tasks.extend([
            asyncio.create_task(self._source_monitor()),
            asyncio.create_task(self._call_processor()),
            asyncio.create_task(self._stats_updater())
        ])
        
        logger.info(f"Call Acquisition Agent initialized with {len(self.call_sources)} sources")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        logger.info("Cleaning up Call Acquisition Agent")
        
        # Save call sources configuration
        await self._save_call_sources()
        
        # Process remaining calls in queue
        while not self.processing_queue.empty():
            try:
                call_data = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                await self._process_acquired_call(call_data)
            except asyncio.TimeoutError:
                break
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process acquisition tasks."""
        task_type = task.get("type")
        task_data = task.get("data", {})
        
        if task_type == "check_source":
            return await self._check_single_source(task_data["source_id"])
        elif task_type == "process_call":
            return await self._process_acquired_call(task_data)
        elif task_type == "add_source":
            return await self._add_call_source(task_data)
        elif task_type == "health_check":
            return await self._perform_health_check()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def add_call_source(
        self,
        source_type: str,
        name: str,
        config: Dict[str, Any],
        enabled: bool = True,
        polling_interval: int = 300
    ) -> str:
        """
        Add a new call source.
        
        Args:
            source_type: Type of source (file_system, ftp, api, email, webhook)
            name: Human-readable name for the source
            config: Source-specific configuration
            enabled: Whether the source is enabled
            polling_interval: Polling interval in seconds
            
        Returns:
            Source ID
        """
        source_id = f"{source_type}_{len(self.call_sources)}"
        
        source = CallSource(
            source_id=source_id,
            source_type=source_type,
            name=name,
            config=config,
            enabled=enabled,
            polling_interval=polling_interval
        )
        
        # Validate source configuration
        if await self._validate_source_config(source):
            self.call_sources[source_id] = source
            self.stats["sources_monitored"] = len([s for s in self.call_sources.values() if s.enabled])
            
            logger.info(f"Added call source: {name} ({source_id})")
            return source_id
        else:
            raise ValueError(f"Invalid source configuration for {name}")
    
    async def _load_call_sources(self) -> None:
        """Load call sources from configuration."""
        # Default file system source
        if self.settings.data_dir:
            await self.add_call_source(
                source_type="file_system",
                name="Default File System",
                config={
                    "directory": str(self.settings.data_dir / "incoming_calls"),
                    "patterns": ["*.txt", "*.json", "*.wav", "*.mp3"],
                    "recursive": True,
                    "move_processed": True,
                    "processed_dir": str(self.settings.data_dir / "processed_calls")
                }
            )
        
        # Load additional sources from config file if exists
        config_file = self.settings.data_dir / "call_sources.json"
        if config_file.exists():
            try:
                async with aiofiles.open(config_file, 'r') as f:
                    content = await f.read()
                    sources_config = json.loads(content)
                
                for source_config in sources_config.get("sources", []):
                    await self.add_call_source(**source_config)
                    
            except Exception as e:
                logger.error(f"Error loading call sources config: {e}")
    
    async def _save_call_sources(self) -> None:
        """Save call sources configuration."""
        config_file = self.settings.data_dir / "call_sources.json"
        
        sources_config = {
            "sources": [
                {
                    "source_type": source.source_type,
                    "name": source.name,
                    "config": source.config,
                    "enabled": source.enabled,
                    "polling_interval": source.polling_interval
                }
                for source in self.call_sources.values()
            ]
        }
        
        try:
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(json.dumps(sources_config, indent=2))
        except Exception as e:
            logger.error(f"Error saving call sources config: {e}")
    
    async def _validate_source_config(self, source: CallSource) -> bool:
        """Validate source configuration."""
        try:
            if source.source_type == "file_system":
                directory = Path(source.config.get("directory", ""))
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                return True
                
            elif source.source_type == "ftp":
                # Test FTP connection
                config = source.config
                ftp = FTP()
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: ftp.connect(config["host"], config.get("port", 21))
                )
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ftp.login(config["username"], config["password"])
                )
                ftp.quit()
                return True
                
            elif source.source_type == "api":
                # Test API endpoint
                config = source.config
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        config["endpoint"],
                        headers=config.get("headers", {}),
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        return response.status < 400
                        
            elif source.source_type == "email":
                # Validate email configuration
                config = source.config
                required_fields = ["server", "port", "username", "password"]
                return all(field in config for field in required_fields)
                
            elif source.source_type == "webhook":
                # Webhook sources are always valid (just need endpoint)
                return "endpoint" in source.config
                
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error validating source {source.name}: {e}")
            return False
    
    async def _source_monitor(self) -> None:
        """Monitor all enabled call sources."""
        while self._running:
            try:
                current_time = datetime.now()
                
                for source in self.call_sources.values():
                    if not source.enabled:
                        continue
                    
                    # Check if it's time to poll this source
                    if (source.last_check is None or 
                        (current_time - source.last_check).seconds >= source.polling_interval):
                        
                        # Submit source check task
                        await self.submit_task(
                            "check_source",
                            {"source_id": source.source_id}
                        )
                        
                        source.last_check = current_time
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in source monitor: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_single_source(self, source_id: str) -> Dict[str, Any]:
        """Check a single call source for new calls."""
        if source_id not in self.call_sources:
            return {"error": f"Source {source_id} not found"}
        
        source = self.call_sources[source_id]
        new_calls = []
        
        try:
            if source.source_type == "file_system":
                new_calls = await self._check_file_system_source(source)
            elif source.source_type == "ftp":
                new_calls = await self._check_ftp_source(source)
            elif source.source_type == "api":
                new_calls = await self._check_api_source(source)
            elif source.source_type == "email":
                new_calls = await self._check_email_source(source)
            
            # Queue new calls for processing
            for call_data in new_calls:
                call_data["source_id"] = source_id
                call_data["acquired_at"] = datetime.now()
                await self.processing_queue.put(call_data)
            
            # Update statistics
            source.total_calls_acquired += len(new_calls)
            self.stats["successful_acquisitions"] += len(new_calls)
            
            logger.debug(f"Source {source.name} acquired {len(new_calls)} new calls")
            
            return {
                "source_id": source_id,
                "new_calls": len(new_calls),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error checking source {source.name}: {e}")
            source.errors.append({
                "timestamp": datetime.now(),
                "error": str(e)
            })
            self.stats["failed_acquisitions"] += 1
            
            return {
                "source_id": source_id,
                "error": str(e),
                "status": "error"
            }
    
    async def _check_file_system_source(self, source: CallSource) -> List[Dict[str, Any]]:
        """Check file system source for new calls."""
        config = source.config
        directory = Path(config["directory"])
        patterns = config.get("patterns", ["*.txt", "*.json"])
        
        new_calls = []
        
        if not directory.exists():
            return new_calls
        
        for pattern in patterns:
            if config.get("recursive", False):
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            
            for file_path in files:
                if file_path.is_file():
                    # Check if file is new (modified in last polling interval)
                    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if (datetime.now() - modified_time).seconds < source.polling_interval * 2:
                        
                        call_data = await self._extract_call_from_file(file_path)
                        if call_data:
                            call_data["source_file"] = str(file_path)
                            new_calls.append(call_data)
                            
                            # Move file if configured
                            if config.get("move_processed", False):
                                processed_dir = Path(config.get("processed_dir", directory / "processed"))
                                processed_dir.mkdir(parents=True, exist_ok=True)
                                
                                new_path = processed_dir / file_path.name
                                file_path.rename(new_path)
        
        return new_calls
    
    async def _check_ftp_source(self, source: CallSource) -> List[Dict[str, Any]]:
        """Check FTP source for new calls."""
        config = source.config
        new_calls = []
        
        def ftp_check():
            nonlocal new_calls
            try:
                ftp = FTP()
                ftp.connect(config["host"], config.get("port", 21))
                ftp.login(config["username"], config["password"])
                
                # Change to target directory
                if "directory" in config:
                    ftp.cwd(config["directory"])
                
                # List files
                files = []
                ftp.retrlines('LIST', files.append)
                
                # Process recent files
                for file_info in files:
                    # Simple parsing - in production, use more robust parsing
                    parts = file_info.split()
                    if len(parts) >= 9:
                        filename = parts[-1]
                        
                        # Check file extension
                        if any(filename.endswith(ext) for ext in config.get("extensions", [".txt", ".wav"])):
                            # Download file content
                            content = []
                            ftp.retrlines(f'RETR {filename}', content.append)
                            
                            call_data = {
                                "call_id": f"ftp_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                "transcript": '\n'.join(content) if filename.endswith('.txt') else "",
                                "filename": filename,
                                "file_type": "text" if filename.endswith('.txt') else "audio"
                            }
                            new_calls.append(call_data)
                
                ftp.quit()
                
            except Exception as e:
                logger.error(f"FTP check error: {e}")
                raise
        
        # Run FTP check in thread pool
        await asyncio.get_event_loop().run_in_executor(None, ftp_check)
        
        return new_calls
    
    async def _check_api_source(self, source: CallSource) -> List[Dict[str, Any]]:
        """Check API source for new calls."""
        config = source.config
        new_calls = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Build request
                url = config["endpoint"]
                headers = config.get("headers", {})
                params = config.get("params", {})
                
                # Add timestamp parameter if configured
                if config.get("use_timestamp_filter", False):
                    last_check = source.last_check or (datetime.now() - timedelta(minutes=source.polling_interval))
                    params["since"] = last_check.isoformat()
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract calls from response
                        calls_data = data.get("calls", []) if isinstance(data, dict) else data
                        
                        for call_item in calls_data:
                            call_data = {
                                "call_id": call_item.get("id", f"api_call_{datetime.now().timestamp()}"),
                                "transcript": call_item.get("transcript", ""),
                                "timestamp": call_item.get("timestamp"),
                                "duration": call_item.get("duration"),
                                "caller_info": call_item.get("caller", {}),
                                "api_data": call_item
                            }
                            new_calls.append(call_data)
                    
                    else:
                        logger.warning(f"API returned status {response.status}")
        
        except Exception as e:
            logger.error(f"API check error: {e}")
            raise
        
        return new_calls
    
    async def _check_email_source(self, source: CallSource) -> List[Dict[str, Any]]:
        """Check email source for new calls."""
        # This would integrate with email libraries like imaplib
        # For now, return empty list
        logger.debug("Email source checking not yet implemented")
        return []
    
    async def _extract_call_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract call data from a file."""
        try:
            if file_path.suffix.lower() == '.json':
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            
            elif file_path.suffix.lower() == '.txt':
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    return {
                        "call_id": file_path.stem,
                        "transcript": content,
                        "file_type": "text"
                    }
            
            elif file_path.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                # Audio file - would need speech-to-text processing
                return {
                    "call_id": file_path.stem,
                    "audio_file": str(file_path),
                    "file_type": "audio",
                    "needs_transcription": True
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting call from file {file_path}: {e}")
            return None
    
    async def _call_processor(self) -> None:
        """Process acquired calls."""
        while self._running:
            try:
                # Get call from processing queue
                call_data = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                # Submit call processing task
                await self.submit_task("process_call", call_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in call processor: {e}")
    
    async def _process_acquired_call(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single acquired call."""
        try:
            call_id = call_data.get("call_id", f"unknown_{datetime.now().timestamp()}")
            
            # Validate call data
            if not call_data.get("transcript") and not call_data.get("needs_transcription"):
                return {"error": "No transcript or audio file provided", "call_id": call_id}
            
            # Handle audio transcription if needed
            if call_data.get("needs_transcription") and call_data.get("audio_file"):
                # This would integrate with speech-to-text service
                logger.info(f"Audio transcription needed for call {call_id}")
                # For now, skip audio processing
                return {"status": "skipped", "reason": "Audio transcription not implemented", "call_id": call_id}
            
            # Send call for analysis
            await self.send_message(
                "analysis_orchestrator",
                "new_call",
                {
                    "call_id": call_id,
                    "call_data": call_data,
                    "source": "call_acquisition_agent"
                },
                MessageType.NOTIFICATION,
                priority=2
            )
            
            # Update statistics
            self.stats["total_calls_acquired"] += 1
            self.stats["last_acquisition"] = datetime.now()
            
            # Store acquired call
            self.acquired_calls.append({
                "call_id": call_id,
                "acquired_at": datetime.now(),
                "source_id": call_data.get("source_id"),
                "processed": True
            })
            
            # Keep only recent acquisitions in memory
            if len(self.acquired_calls) > 1000:
                self.acquired_calls = self.acquired_calls[-500:]
            
            logger.info(f"Successfully processed acquired call {call_id}")
            
            return {"status": "success", "call_id": call_id}
            
        except Exception as e:
            logger.error(f"Error processing acquired call: {e}")
            return {"error": str(e), "call_id": call_data.get("call_id", "unknown")}
    
    async def _stats_updater(self) -> None:
        """Update acquisition statistics."""
        while self._running:
            try:
                # Update source statistics
                enabled_sources = [s for s in self.call_sources.values() if s.enabled]
                self.stats["sources_monitored"] = len(enabled_sources)
                
                # Clean up old errors (keep only last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                for source in self.call_sources.values():
                    source.errors = [
                        error for error in source.errors
                        if error["timestamp"] > cutoff_time
                    ]
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform health check on all sources."""
        health_status = {
            "overall_status": "healthy",
            "sources": {},
            "issues": []
        }
        
        for source_id, source in self.call_sources.items():
            source_status = {
                "enabled": source.enabled,
                "last_check": source.last_check.isoformat() if source.last_check else None,
                "total_calls": source.total_calls_acquired,
                "recent_errors": len([e for e in source.errors if (datetime.now() - e["timestamp"]).hours < 1])
            }
            
            # Check if source is healthy
            if source.enabled:
                if source.last_check is None:
                    source_status["status"] = "not_checked"
                elif (datetime.now() - source.last_check).seconds > source.polling_interval * 2:
                    source_status["status"] = "stale"
                    health_status["issues"].append(f"Source {source.name} has stale data")
                elif len(source.errors) > 5:  # Too many recent errors
                    source_status["status"] = "error"
                    health_status["issues"].append(f"Source {source.name} has many errors")
                else:
                    source_status["status"] = "healthy"
            else:
                source_status["status"] = "disabled"
            
            health_status["sources"][source_id] = source_status
        
        # Update overall status
        if health_status["issues"]:
            health_status["overall_status"] = "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
        
        return health_status
    
    # Message handlers
    
    async def _handle_add_source(self, message) -> None:
        """Handle add source request."""
        try:
            payload = message.payload
            source_id = await self.add_call_source(**payload)
            
            await self.send_message(
                message.sender,
                "source_added",
                {"source_id": source_id, "status": "success"},
                MessageType.RESPONSE
            )
        except Exception as e:
            await self.send_message(
                message.sender,
                "source_add_error",
                {"error": str(e)},
                MessageType.ERROR
            )
    
    async def _handle_remove_source(self, message) -> None:
        """Handle remove source request."""
        source_id = message.payload.get("source_id")
        
        if source_id in self.call_sources:
            del self.call_sources[source_id]
            await self.send_message(
                message.sender,
                "source_removed",
                {"source_id": source_id, "status": "success"},
                MessageType.RESPONSE
            )
        else:
            await self.send_message(
                message.sender,
                "source_not_found",
                {"source_id": source_id, "error": "Source not found"},
                MessageType.ERROR
            )
    
    async def _handle_list_sources(self, message) -> None:
        """Handle list sources request."""
        sources = []
        for source_id, source in self.call_sources.items():
            sources.append({
                "source_id": source_id,
                "name": source.name,
                "type": source.source_type,
                "enabled": source.enabled,
                "last_check": source.last_check.isoformat() if source.last_check else None,
                "total_calls": source.total_calls_acquired,
                "errors": len(source.errors)
            })
        
        await self.send_message(
            message.sender,
            "sources_list",
            {"sources": sources},
            MessageType.RESPONSE
        )
    
    async def _handle_get_stats(self, message) -> None:
        """Handle get statistics request."""
        await self.send_message(
            message.sender,
            "acquisition_stats",
            self.stats,
            MessageType.RESPONSE
        )
    
    async def _handle_manual_check(self, message) -> None:
        """Handle manual source check request."""
        source_id = message.payload.get("source_id")
        
        if source_id:
            result = await self._check_single_source(source_id)
        else:
            # Check all sources
            results = []
            for source_id in self.call_sources.keys():
                result = await self._check_single_source(source_id)
                results.append(result)
            result = {"results": results}
        
        await self.send_message(
            message.sender,
            "manual_check_result",
            result,
            MessageType.RESPONSE
        )