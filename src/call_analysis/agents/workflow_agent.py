"""
Workflow Agent for managing complex multi-step analysis workflows.

This agent orchestrates sophisticated workflows involving multiple agents,
handles conditional logic, manages workflow state, and provides
workflow automation capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json
from pathlib import Path
import uuid

from .base_agent import BaseAgent, AgentStatus, AgentCapability, MessageType
from ..config import get_settings

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class WorkflowStep:
    """Represents a single step in a workflow."""
    
    def __init__(
        self,
        step_id: str,
        name: str,
        agent_id: str,
        action: str,
        parameters: Dict[str, Any],
        conditions: Optional[Dict[str, Any]] = None,
        retry_count: int = 3,
        timeout: int = 300
    ):
        self.step_id = step_id
        self.name = name
        self.agent_id = agent_id
        self.action = action
        self.parameters = parameters
        self.conditions = conditions or {}
        self.retry_count = retry_count
        self.timeout = timeout
        self.status = WorkflowStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.attempts = 0


class Workflow:
    """Represents a complete workflow definition."""
    
    def __init__(
        self,
        workflow_id: str,
        name: str,
        description: str,
        steps: List[WorkflowStep],
        triggers: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps = steps
        self.triggers = triggers or {}
        self.metadata = metadata or {}
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.current_step = 0
        self.context = {}
        self.results = {}


class WorkflowAgent(BaseAgent):
    """
    Manages and executes complex multi-step workflows.
    
    Provides workflow definition, execution, monitoring, and
    orchestration capabilities across multiple agents.
    """
    
    def __init__(
        self,
        agent_id: str = "workflow_agent",
        name: str = "Workflow Agent"
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="Manages complex multi-step analysis workflows",
            max_concurrent_tasks=20
        )
        
        self.settings = get_settings()
        
        # Workflow management
        self.workflows: Dict[str, Workflow] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # Execution queue
        self.execution_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "active_workflows": 0,
            "average_execution_time": 0.0,
            "last_execution": None
        }
        
        # Setup capabilities
        self._setup_capabilities()
        
        # Load predefined workflows
        self._load_predefined_workflows()
        
        # Register message handlers
        self.register_handler("execute_workflow", self._handle_execute_workflow)
        self.register_handler("create_workflow", self._handle_create_workflow)
        self.register_handler("get_workflow_status", self._handle_get_workflow_status)
        self.register_handler("pause_workflow", self._handle_pause_workflow)
        self.register_handler("resume_workflow", self._handle_resume_workflow)
        self.register_handler("cancel_workflow", self._handle_cancel_workflow)
        self.register_handler("workflow_step_completed", self._handle_step_completion)
    
    def _setup_capabilities(self):
        """Setup workflow capabilities."""
        capabilities = [
            AgentCapability(
                name="workflow_execution",
                description="Execute multi-step analysis workflows",
                input_types=["workflow_definition", "execution_context"],
                output_types=["workflow_results", "execution_status"],
                estimated_duration=timedelta(minutes=10)
            ),
            AgentCapability(
                name="workflow_orchestration",
                description="Orchestrate multiple agents in complex workflows",
                input_types=["agent_coordination", "step_dependencies"],
                output_types=["orchestration_results"],
                estimated_duration=timedelta(minutes=5)
            ),
            AgentCapability(
                name="conditional_logic",
                description="Handle conditional workflow execution paths",
                input_types=["conditions", "context_data"],
                output_types=["execution_path", "decision_results"],
                estimated_duration=timedelta(seconds=30)
            ),
            AgentCapability(
                name="workflow_monitoring",
                description="Monitor and track workflow execution progress",
                input_types=["execution_state"],
                output_types=["progress_reports", "status_updates"],
                estimated_duration=timedelta(seconds=15)
            ),
            AgentCapability(
                name="error_handling",
                description="Handle workflow errors and implement retry logic",
                input_types=["error_conditions", "retry_policies"],
                output_types=["recovery_actions", "error_reports"],
                estimated_duration=timedelta(seconds=45)
            )
        ]
        
        for capability in capabilities:
            self.add_capability(capability)
    
    def _load_predefined_workflows(self):
        """Load predefined workflow templates."""
        self.workflow_templates = {
            "comprehensive_call_analysis": {
                "name": "Comprehensive Call Analysis",
                "description": "Complete analysis pipeline for incoming calls",
                "steps": [
                    {
                        "step_id": "acquire_call",
                        "name": "Acquire Call Data",
                        "agent_id": "call_acquisition_agent",
                        "action": "process_call",
                        "parameters": {"call_data": "{input.call_data}"},
                        "timeout": 60
                    },
                    {
                        "step_id": "analyze_call",
                        "name": "Comprehensive Analysis",
                        "agent_id": "analysis_orchestrator",
                        "action": "analyze_call",
                        "parameters": {
                            "call_id": "{step.acquire_call.result.call_id}",
                            "transcript": "{step.acquire_call.result.transcript}"
                        },
                        "conditions": {
                            "require": ["step.acquire_call.status == 'success'"]
                        },
                        "timeout": 180
                    },
                    {
                        "step_id": "generate_report",
                        "name": "Generate Analysis Report",
                        "agent_id": "workflow_agent",
                        "action": "generate_report",
                        "parameters": {
                            "analysis_results": "{step.analyze_call.result}",
                            "call_id": "{step.acquire_call.result.call_id}"
                        },
                        "conditions": {
                            "require": ["step.analyze_call.status == 'success'"]
                        },
                        "timeout": 60
                    }
                ],
                "triggers": {
                    "new_call": True,
                    "manual": True
                }
            },
            
            "batch_analysis_workflow": {
                "name": "Batch Call Analysis",
                "description": "Process multiple calls in batch",
                "steps": [
                    {
                        "step_id": "prepare_batch",
                        "name": "Prepare Batch Processing",
                        "agent_id": "workflow_agent",
                        "action": "prepare_batch",
                        "parameters": {"call_ids": "{input.call_ids}"},
                        "timeout": 30
                    },
                    {
                        "step_id": "batch_analyze",
                        "name": "Batch Analysis",
                        "agent_id": "analysis_orchestrator",
                        "action": "batch_analyze",
                        "parameters": {"calls": "{step.prepare_batch.result.calls}"},
                        "timeout": 600
                    },
                    {
                        "step_id": "aggregate_results",
                        "name": "Aggregate Results",
                        "agent_id": "workflow_agent",
                        "action": "aggregate_results",
                        "parameters": {"batch_results": "{step.batch_analyze.result}"},
                        "timeout": 120
                    }
                ]
            },
            
            "quality_monitoring_workflow": {
                "name": "Quality Monitoring",
                "description": "Monitor and assess call quality",
                "steps": [
                    {
                        "step_id": "quality_check",
                        "name": "Perform Quality Assessment",
                        "agent_id": "analysis_orchestrator",
                        "action": "quality_assessment",
                        "parameters": {
                            "call_id": "{input.call_id}",
                            "transcript": "{input.transcript}"
                        },
                        "timeout": 90
                    },
                    {
                        "step_id": "generate_feedback",
                        "name": "Generate Quality Feedback",
                        "agent_id": "workflow_agent",
                        "action": "generate_feedback",
                        "parameters": {"assessment": "{step.quality_check.result}"},
                        "conditions": {
                            "require": ["step.quality_check.result.quality_score < 0.7"]
                        },
                        "timeout": 60
                    },
                    {
                        "step_id": "escalate_if_needed",
                        "name": "Escalate Poor Quality Calls",
                        "agent_id": "monitoring_agent",
                        "action": "create_alert",
                        "parameters": {
                            "severity": "medium",
                            "message": "Poor quality call detected: {input.call_id}",
                            "component": "quality_monitoring"
                        },
                        "conditions": {
                            "require": ["step.quality_check.result.quality_score < 0.5"]
                        },
                        "timeout": 30
                    }
                ]
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the workflow agent."""
        logger.info("Initializing Workflow Agent")
        
        # Load existing workflows
        await self._load_workflows()
        
        # Start workflow processors
        self._background_tasks.extend([
            asyncio.create_task(self._workflow_executor()),
            asyncio.create_task(self._workflow_monitor()),
            asyncio.create_task(self._stats_updater())
        ])
        
        logger.info("Workflow Agent initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup workflow agent resources."""
        logger.info("Cleaning up Workflow Agent")
        
        # Save workflow state
        await self._save_workflows()
        
        # Cancel active workflows gracefully
        for workflow_id in list(self.active_executions.keys()):
            await self._cancel_workflow(workflow_id, "System shutdown")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process workflow tasks."""
        task_type = task.get("type")
        task_data = task.get("data", {})
        
        if task_type == "execute_workflow":
            return await self._execute_workflow_task(task_data)
        elif task_type == "create_workflow":
            return await self._create_workflow_task(task_data)
        elif task_type == "monitor_workflow":
            return await self._monitor_workflow_task(task_data)
        elif task_type == "generate_report":
            return await self._generate_workflow_report(task_data)
        elif task_type == "prepare_batch":
            return await self._prepare_batch_task(task_data)
        elif task_type == "aggregate_results":
            return await self._aggregate_results_task(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def create_workflow(
        self,
        template_name: str,
        input_data: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> str:
        """
        Create a new workflow instance from template.
        
        Args:
            template_name: Name of the workflow template
            input_data: Input data for the workflow
            workflow_id: Optional custom workflow ID
            
        Returns:
            Workflow ID
        """
        if template_name not in self.workflow_templates:
            raise ValueError(f"Unknown workflow template: {template_name}")
        
        workflow_id = workflow_id or f"workflow_{uuid.uuid4().hex[:8]}"
        template = self.workflow_templates[template_name]
        
        # Create workflow steps
        steps = []
        for step_def in template["steps"]:
            step = WorkflowStep(
                step_id=step_def["step_id"],
                name=step_def["name"],
                agent_id=step_def["agent_id"],
                action=step_def["action"],
                parameters=step_def["parameters"],
                conditions=step_def.get("conditions"),
                retry_count=step_def.get("retry_count", 3),
                timeout=step_def.get("timeout", 300)
            )
            steps.append(step)
        
        # Create workflow
        workflow = Workflow(
            workflow_id=workflow_id,
            name=template["name"],
            description=template["description"],
            steps=steps,
            triggers=template.get("triggers"),
            metadata={"template": template_name, "input_data": input_data}
        )
        
        workflow.context.update(input_data)
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id} from template {template_name}")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            Execution result
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status not in [WorkflowStatus.PENDING, WorkflowStatus.PAUSED]:
            raise ValueError(f"Workflow {workflow_id} is not in executable state: {workflow.status}")
        
        # Submit for execution
        await self.execution_queue.put({
            "workflow_id": workflow_id,
            "action": "execute"
        })
        
        return {"status": "submitted", "workflow_id": workflow_id}
    
    async def _execute_workflow_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow task."""
        workflow_id = task_data["workflow_id"]
        workflow = self.workflows[workflow_id]
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        
        self.active_executions[workflow_id] = {
            "workflow": workflow,
            "start_time": datetime.now(),
            "current_step": workflow.current_step
        }
        
        try:
            # Execute workflow steps
            result = await self._execute_workflow_steps(workflow)
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            # Update statistics
            self.stats["successful_workflows"] += 1
            execution_time = (datetime.now() - workflow.started_at).total_seconds()
            self._update_average_execution_time(execution_time)
            
            logger.info(f"Workflow {workflow_id} completed successfully in {execution_time:.2f}s")
            
            return {
                "status": "completed",
                "workflow_id": workflow_id,
                "results": workflow.results,
                "execution_time": execution_time
            }
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            
            self.stats["failed_workflows"] += 1
            
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
            return {
                "status": "failed",
                "workflow_id": workflow_id,
                "error": str(e)
            }
        
        finally:
            if workflow_id in self.active_executions:
                del self.active_executions[workflow_id]
    
    async def _execute_workflow_steps(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute individual workflow steps."""
        results = {}
        
        for i, step in enumerate(workflow.steps[workflow.current_step:], workflow.current_step):
            workflow.current_step = i
            
            # Check step conditions
            if not await self._check_step_conditions(step, workflow):
                logger.info(f"Skipping step {step.step_id} due to unmet conditions")
                continue
            
            # Execute step with retries
            step_result = await self._execute_step_with_retry(step, workflow)
            
            # Store step result
            results[step.step_id] = step_result
            workflow.results[step.step_id] = step_result
            
            # Update workflow context
            workflow.context[f"step.{step.step_id}"] = {
                "result": step_result,
                "status": step.status.value
            }
            
            # Check if step failed and should stop workflow
            if step.status == WorkflowStatus.FAILED and not step.conditions.get("continue_on_failure", False):
                raise Exception(f"Step {step.step_id} failed: {step.error}")
        
        return results
    
    async def _execute_step_with_retry(self, step: WorkflowStep, workflow: Workflow) -> Dict[str, Any]:
        """Execute a single step with retry logic."""
        step.started_at = datetime.now()
        step.attempts = 0
        
        while step.attempts < step.retry_count:
            step.attempts += 1
            
            try:
                # Resolve parameters
                resolved_params = self._resolve_parameters(step.parameters, workflow)
                
                # Execute step
                result = await self._execute_single_step(
                    step.agent_id,
                    step.action,
                    resolved_params,
                    step.timeout
                )
                
                step.status = WorkflowStatus.COMPLETED
                step.completed_at = datetime.now()
                step.result = result
                
                logger.debug(f"Step {step.step_id} completed successfully")
                return result
                
            except Exception as e:
                step.error = str(e)
                logger.warning(f"Step {step.step_id} attempt {step.attempts} failed: {e}")
                
                if step.attempts < step.retry_count:
                    await asyncio.sleep(min(2 ** step.attempts, 30))  # Exponential backoff
        
        # All retries failed
        step.status = WorkflowStatus.FAILED
        step.completed_at = datetime.now()
        
        raise Exception(f"Step {step.step_id} failed after {step.retry_count} attempts: {step.error}")
    
    async def _execute_single_step(
        self,
        agent_id: str,
        action: str,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        # Send message to target agent
        message_id = await self.send_message(
            agent_id,
            action,
            parameters,
            MessageType.COMMAND,
            priority=3,
            requires_response=True
        )
        
        # Wait for response with timeout
        try:
            # This is simplified - in a real implementation, you'd have a response handler
            await asyncio.sleep(1)  # Simulate step execution
            
            return {
                "status": "success",
                "message_id": message_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            raise Exception(f"Step execution timed out after {timeout} seconds")
    
    def _resolve_parameters(self, parameters: Dict[str, Any], workflow: Workflow) -> Dict[str, Any]:
        """Resolve parameter templates with workflow context."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Template parameter
                template = value[1:-1]  # Remove braces
                resolved_value = self._resolve_template(template, workflow)
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        return resolved
    
    def _resolve_template(self, template: str, workflow: Workflow) -> Any:
        """Resolve a template string with workflow context."""
        try:
            # Simple template resolution
            if template.startswith("input."):
                key = template[6:]  # Remove "input."
                return workflow.metadata.get("input_data", {}).get(key)
            elif template.startswith("step."):
                # Extract step reference
                parts = template.split(".")
                if len(parts) >= 3:
                    step_id = parts[1]
                    field = ".".join(parts[2:])
                    step_data = workflow.context.get(f"step.{step_id}", {})
                    
                    # Navigate nested fields
                    current = step_data
                    for field_part in field.split("."):
                        if isinstance(current, dict):
                            current = current.get(field_part)
                        else:
                            return None
                    
                    return current
            
            return template  # Return as-is if not a template
            
        except Exception as e:
            logger.warning(f"Error resolving template '{template}': {e}")
            return template
    
    async def _check_step_conditions(self, step: WorkflowStep, workflow: Workflow) -> bool:
        """Check if step conditions are met."""
        if not step.conditions:
            return True
        
        try:
            required_conditions = step.conditions.get("require", [])
            
            for condition in required_conditions:
                if not self._evaluate_condition(condition, workflow):
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking step conditions: {e}")
            return False
    
    def _evaluate_condition(self, condition: str, workflow: Workflow) -> bool:
        """Evaluate a condition string."""
        try:
            # Simple condition evaluation
            # In a production system, you'd use a proper expression evaluator
            
            if "==" in condition:
                left, right = condition.split("==", 1)
                left_val = self._resolve_template(left.strip(), workflow)
                right_val = right.strip().strip("'\"")
                return str(left_val) == right_val
            
            # Add more condition types as needed
            return True
            
        except Exception as e:
            logger.warning(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _workflow_executor(self) -> None:
        """Execute workflows from the queue."""
        while self._running:
            try:
                # Get workflow execution request
                request = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                
                # Submit execution task
                await self.submit_task("execute_workflow", request)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in workflow executor: {e}")
    
    async def _workflow_monitor(self) -> None:
        """Monitor active workflow executions."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Check for stalled workflows
                for workflow_id, execution_data in list(self.active_executions.items()):
                    start_time = execution_data["start_time"]
                    if (current_time - start_time).seconds > 3600:  # 1 hour timeout
                        logger.warning(f"Workflow {workflow_id} has been running for over 1 hour")
                        
                        # Send alert
                        await self.send_message(
                            "monitoring_agent",
                            "workflow_timeout_alert",
                            {
                                "workflow_id": workflow_id,
                                "runtime": (current_time - start_time).seconds
                            },
                            MessageType.NOTIFICATION,
                            priority=3
                        )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in workflow monitor: {e}")
                await asyncio.sleep(60)
    
    async def _stats_updater(self) -> None:
        """Update workflow statistics."""
        while self._running:
            try:
                self.stats["total_workflows"] = len(self.workflows)
                self.stats["active_workflows"] = len(self.active_executions)
                
                if self.stats["successful_workflows"] > 0:
                    self.stats["last_execution"] = datetime.now().isoformat()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating workflow stats: {e}")
                await asyncio.sleep(30)
    
    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update average workflow execution time."""
        total_workflows = self.stats["successful_workflows"]
        if total_workflows == 1:
            self.stats["average_execution_time"] = execution_time
        else:
            current_avg = self.stats["average_execution_time"]
            self.stats["average_execution_time"] = (
                (current_avg * (total_workflows - 1) + execution_time) / total_workflows
            )
    
    async def _cancel_workflow(self, workflow_id: str, reason: str = "User cancelled") -> None:
        """Cancel an active workflow."""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            
            if workflow_id in self.active_executions:
                del self.active_executions[workflow_id]
            
            logger.info(f"Workflow {workflow_id} cancelled: {reason}")
    
    async def _load_workflows(self) -> None:
        """Load existing workflows from disk."""
        try:
            workflows_file = self.settings.data_dir / "workflows.json"
            if workflows_file.exists():
                with open(workflows_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct workflows (simplified)
                logger.info(f"Loaded {len(data)} workflows from disk")
        
        except Exception as e:
            logger.error(f"Error loading workflows: {e}")
    
    async def _save_workflows(self) -> None:
        """Save workflows to disk."""
        try:
            workflows_file = self.settings.data_dir / "workflows.json"
            
            # Serialize workflows (simplified)
            workflow_data = []
            for workflow in self.workflows.values():
                workflow_data.append({
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "status": workflow.status.value,
                    "created_at": workflow.created_at.isoformat(),
                    "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
                })
            
            with open(workflows_file, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            
            logger.info(f"Saved {len(workflow_data)} workflows")
        
        except Exception as e:
            logger.error(f"Error saving workflows: {e}")
    
    # Task implementations
    
    async def _prepare_batch_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch processing task."""
        call_ids = task_data.get("call_ids", [])
        
        # Simulate batch preparation
        calls = [{"call_id": call_id, "status": "ready"} for call_id in call_ids]
        
        return {
            "status": "success",
            "calls": calls,
            "batch_size": len(calls)
        }
    
    async def _aggregate_results_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate batch results task."""
        batch_results = task_data.get("batch_results", {})
        
        # Simulate result aggregation
        summary = {
            "total_calls": len(batch_results.get("results", [])),
            "successful": 0,
            "failed": 0,
            "aggregated_insights": []
        }
        
        return {
            "status": "success",
            "summary": summary
        }
    
    async def _generate_workflow_report(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate workflow execution report."""
        analysis_results = task_data.get("analysis_results", {})
        call_id = task_data.get("call_id")
        
        report = {
            "call_id": call_id,
            "report_generated_at": datetime.now().isoformat(),
            "analysis_summary": analysis_results,
            "report_type": "comprehensive_analysis"
        }
        
        return {
            "status": "success",
            "report": report
        }
    
    # Message handlers
    
    async def _handle_execute_workflow(self, message) -> None:
        """Handle workflow execution request."""
        try:
            payload = message.payload
            workflow_id = payload.get("workflow_id")
            
            if not workflow_id:
                template_name = payload.get("template_name")
                input_data = payload.get("input_data", {})
                workflow_id = await self.create_workflow(template_name, input_data)
            
            result = await self.execute_workflow(workflow_id)
            
            await self.send_message(
                message.sender,
                "workflow_execution_response",
                result,
                MessageType.RESPONSE
            )
            
        except Exception as e:
            await self.send_message(
                message.sender,
                "workflow_execution_error",
                {"error": str(e)},
                MessageType.ERROR
            )
    
    async def _handle_create_workflow(self, message) -> None:
        """Handle workflow creation request."""
        try:
            payload = message.payload
            template_name = payload.get("template_name")
            input_data = payload.get("input_data", {})
            
            workflow_id = await self.create_workflow(template_name, input_data)
            
            await self.send_message(
                message.sender,
                "workflow_created",
                {"workflow_id": workflow_id, "status": "created"},
                MessageType.RESPONSE
            )
            
        except Exception as e:
            await self.send_message(
                message.sender,
                "workflow_creation_error",
                {"error": str(e)},
                MessageType.ERROR
            )
    
    async def _handle_get_workflow_status(self, message) -> None:
        """Handle workflow status request."""
        workflow_id = message.payload.get("workflow_id")
        
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            status = {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "current_step": workflow.current_step,
                "total_steps": len(workflow.steps),
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
            }
        else:
            status = {"error": f"Workflow {workflow_id} not found"}
        
        await self.send_message(
            message.sender,
            "workflow_status_response",
            status,
            MessageType.RESPONSE
        )
    
    async def _handle_pause_workflow(self, message) -> None:
        """Handle workflow pause request."""
        workflow_id = message.payload.get("workflow_id")
        
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.PAUSED
                response = {"status": "paused", "workflow_id": workflow_id}
            else:
                response = {"error": f"Workflow {workflow_id} is not running"}
        else:
            response = {"error": f"Workflow {workflow_id} not found"}
        
        await self.send_message(
            message.sender,
            "workflow_pause_response",
            response,
            MessageType.RESPONSE
        )
    
    async def _handle_resume_workflow(self, message) -> None:
        """Handle workflow resume request."""
        workflow_id = message.payload.get("workflow_id")
        
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            if workflow.status == WorkflowStatus.PAUSED:
                await self.execute_workflow(workflow_id)
                response = {"status": "resumed", "workflow_id": workflow_id}
            else:
                response = {"error": f"Workflow {workflow_id} is not paused"}
        else:
            response = {"error": f"Workflow {workflow_id} not found"}
        
        await self.send_message(
            message.sender,
            "workflow_resume_response",
            response,
            MessageType.RESPONSE
        )
    
    async def _handle_cancel_workflow(self, message) -> None:
        """Handle workflow cancellation request."""
        workflow_id = message.payload.get("workflow_id")
        reason = message.payload.get("reason", "User cancelled")
        
        await self._cancel_workflow(workflow_id, reason)
        
        await self.send_message(
            message.sender,
            "workflow_cancel_response",
            {"status": "cancelled", "workflow_id": workflow_id},
            MessageType.RESPONSE
        )
    
    async def _handle_step_completion(self, message) -> None:
        """Handle workflow step completion notification."""
        # This would be used in a more sophisticated implementation
        # where agents report back step completion status
        pass