"""
🎯 Agent Coordinator - AI-Powered Orchestration Layer for AgriPal AI Agents
Intelligent coordinator using OpenAI Agents SDK to orchestrate communication and workflow 
between PerceptionAgent, KnowledgeAgent, and EmailAgent.
"""
import asyncio
import json
import logging
import re
import random
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

# OpenAI imports
from openai import AsyncOpenAI
from agents import Agent, Runner, RunResult

from ..agripal_agents.perception_agent import PerceptionAgent
from ..agripal_agents.knowledge_agent import KnowledgeAgent
from ..agripal_agents.email_agent import EmailAgent
from ..models import (
    AgentMessage, 
    AgentResponse, 
    AgentType,
    SessionMessage
)
from ..config import settings

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Types of agent workflows"""
    FULL_CONSULTATION = "full_consultation"  # All agents in sequence
    IMAGE_ANALYSIS_ONLY = "image_analysis_only"  # Perception only
    KNOWLEDGE_SEARCH_ONLY = "knowledge_search_only"  # Knowledge only
    EMAIL_REPORT_ONLY = "email_report_only"  # Email only
    PERCEPTION_TO_KNOWLEDGE = "perception_to_knowledge"  # Perception → Knowledge
    KNOWLEDGE_TO_EMAIL = "knowledge_to_email"  # Knowledge → Email

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"

class AgentCoordinator:
    """
    🎯 AI-Powered Central Coordinator for orchestrating AgriPal AI agents using OpenAI Agents SDK
    
    Capabilities:
    - Intelligent workflow decision making using AI
    - Dynamic agent selection based on user input
    - Smart error recovery and fallback strategies
    - Adaptive optimization of agent communication
    - Context-aware workflow orchestration
    """
    
    def __init__(self):
        # OpenAI Client and Agent
        self.model = settings.OPENAI_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.coordinator_agent = None
        
        # Initialize all agents
        self.perception_agent = None
        self.knowledge_agent = None
        self.email_agent = None
        
        # Workflow state management
        self.active_workflows = {}
        self.session_cache = {}
        
        # Performance metrics
        self.workflow_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0
        }
        
        # Initialize agents and coordinator lazily (will be called when needed)
        self._initialized = False
        
        logger.info("🎯 AI-Powered Agent Coordinator initialized")
    
    async def _ensure_initialized(self):
        """Ensure all components are initialized"""
        if not self._initialized:
            await self._initialize_all_components()
            self._initialized = True
    
    async def _initialize_all_components(self):
        """
        Initialize all AI agents and the coordinator agent asynchronously
        """
        try:
            logger.info("🔄 Initializing AI agents and coordinator...")
            
            # Initialize agents and coordinator in parallel for faster startup
            initialization_tasks = [
                self._init_perception_agent(),
                self._init_knowledge_agent(), 
                self._init_email_agent(),
                self._setup_coordinator_agent()
            ]
            
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check which agents are ready
            ready_agents = []
            if self.perception_agent: ready_agents.append("Perception")
            if self.knowledge_agent: ready_agents.append("Knowledge") 
            if self.email_agent: ready_agents.append("Email")
            if self.coordinator_agent: ready_agents.append("Coordinator")
            
            logger.info(f"✅ Agent Coordinator ready. Available agents: {', '.join(ready_agents)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {str(e)}")
    
    async def _init_perception_agent(self):
        """Initialize Perception Agent"""
        try:
            self.perception_agent = PerceptionAgent()
            logger.info("✅ Perception Agent initialized")
        except Exception as e:
            logger.error(f"❌ Perception Agent initialization failed: {str(e)}")
    
    async def _init_knowledge_agent(self):
        """Initialize Knowledge Agent"""
        try:
            self.knowledge_agent = KnowledgeAgent()
            # Wait for async components to initialize
            await self.knowledge_agent.initialize()
            logger.info("✅ Knowledge Agent initialized")
        except Exception as e:
            logger.error(f"❌ Knowledge Agent initialization failed: {str(e)}")
    
    async def _init_email_agent(self):
        """Initialize Email Agent"""
        try:
            self.email_agent = EmailAgent()
            logger.info("✅ Email Agent initialized")
        except Exception as e:
            logger.error(f"❌ Email Agent initialization failed: {str(e)}")
    
    async def _setup_coordinator_agent(self):
        """
        Initialize the OpenAI Coordinator Agent with workflow orchestration tools
        """
        try:
            # Create the coordinator agent with workflow expertise
            self.coordinator_agent = Agent(
                name="AgriPal Workflow Coordinator",
                instructions="""
                Expert agricultural workflow coordination agent with intelligent decision-making capabilities.
                
                You are the central orchestrator for AgriPal's AI agent ecosystem, responsible for:
                - Analyzing user requests to determine optimal agent workflows
                - Dynamically selecting which agents to execute based on user needs
                - Coordinating data flow between perception, knowledge, and email agents
                - Making intelligent decisions about parallel vs sequential execution
                - Handling errors gracefully with appropriate fallback strategies
                - Optimizing performance based on current system conditions
                
                Your role is to:
                1. Analyze incoming user requests to understand their agricultural needs
                2. Determine the most efficient workflow (which agents to use and in what order)
                3. Execute agents with appropriate data passing and context
                4. Handle any errors or failures with intelligent recovery strategies
                5. Optimize the overall user experience through smart orchestration
                
                Always prioritize:
                - User experience and satisfaction
                - Accurate and helpful agricultural guidance
                - Efficient resource utilization
                - Graceful error handling and recovery
                - Clear communication about workflow status
                
                You have access to perception (image analysis), knowledge (information retrieval), 
                and email (report generation) agents. Use them wisely based on user needs.
                """,
                model=self.model,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_user_request",
                            "description": "Analyze user input to determine which agents are needed and optimal workflow",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_input": {
                                        "type": "object",
                                        "description": "User input data including query, images, etc."
                                    },
                                    "user_context": {
                                        "type": "object",
                                        "description": "User context and session data"
                                    }
                                },
                                "required": ["user_input"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_perception_agent",
                            "description": "Execute the perception agent for image analysis",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "images": {
                                        "type": "array",
                                        "description": "Images to analyze"
                                    },
                                    "analysis_type": {
                                        "type": "string",
                                        "description": "Type of analysis to perform"
                                    },
                                    "crop_type": {
                                        "type": "string",
                                        "description": "Crop type for context"
                                    }
                                },
                                "required": ["images"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_knowledge_agent",
                            "description": "Execute the knowledge agent for information retrieval",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query"
                                    },
                                    "crop_type": {
                                        "type": "string",
                                        "description": "Crop type for context"
                                    },
                                    "perception_results": {
                                        "type": "object",
                                        "description": "Results from perception agent if available"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_email_agent",
                            "description": "Execute the email agent to send reports",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "recipient_email": {
                                        "type": "string",
                                        "description": "Email recipient"
                                    },
                                    "session_data": {
                                        "type": "object",
                                        "description": "Session data for report generation"
                                    },
                                    "analysis_results": {
                                        "type": "object",
                                        "description": "Results from other agents"
                                    }
                                },
                                "required": ["recipient_email", "session_data"]
                            }
                        }
                    }
                ]
            )
            
            logger.info("✅ Coordinator Agent successfully initialized with intelligent orchestration tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Coordinator Agent: {str(e)}")
            self.coordinator_agent = None
    
    async def execute_workflow(
        self, 
        workflow_type: WorkflowType,
        session_id: str,
        user_input: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete agent workflow
        
        Args:
            workflow_type: Type of workflow to execute
            session_id: Unique session identifier
            user_input: User input data (query, images, etc.)
            user_context: User profile and context information
            
        Returns:
            Dictionary containing workflow results and status
        """
        workflow_id = f"{session_id}_{datetime.utcnow().timestamp()}"
        start_time = datetime.utcnow()
        
        # Initialize workflow state
        workflow_state = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "workflow_type": workflow_type.value,
            "status": WorkflowStatus.IN_PROGRESS.value,
            "start_time": start_time,
            "agents_executed": [],
            "results": {},
            "errors": [],
            "metadata": {
                "user_context": user_context or {},
                "user_input": user_input
            }
        }
        
        self.active_workflows[workflow_id] = workflow_state
        
        try:
            # Try to enrich with recent conversation summary for context-awareness
            try:
                from uuid import UUID as _UUID
                from ..database.connection import get_database_manager as _get_db
                from ..database.services import SessionService as _SessionService
                dbm = await _get_db()
                svc = _SessionService(dbm)
                recent_msgs = await svc.get_recent_messages(_UUID(session_id), limit=12)
                
                # Build enhanced summary with agricultural context
                turn_texts = []
                agricultural_context = {"crops": set(), "problems": set(), "solutions": set(), "location": ""}
                
                for m in reversed(recent_msgs):
                    role = "Farmer" if getattr(m, "message_type", "user") == "user" else "AgriPal"
                    content = (getattr(m, "content", "") or "").replace("\n", " ")
                    
                    # Extract agricultural context from messages
                    if role == "Farmer":
                        extracted = self._extract_agricultural_context(content)
                        agricultural_context["crops"].update(extracted["crops"])
                        agricultural_context["problems"].update(extracted["problems"])
                        agricultural_context["solutions"].update(extracted["solutions"])
                        if extracted["location"]:
                            agricultural_context["location"] = extracted["location"]
                    
                    if len(content) > 160:
                        content = content[:157] + "…"
                    turn_texts.append(f"{role}: {content}")
                
                if turn_texts:
                    workflow_state["rolling_summary"] = " | ".join(turn_texts[-8:])
                    
                # Add structured agricultural context to workflow state
                agricultural_context["crops"] = list(agricultural_context["crops"])
                agricultural_context["problems"] = list(agricultural_context["problems"])
                agricultural_context["solutions"] = list(agricultural_context["solutions"])
                workflow_state["agricultural_context"] = agricultural_context
                
            except Exception as _hist_err:
                logger.debug("Database history not available, trying fallback: %s", str(_hist_err))
                # Try fallback in-memory storage
                try:
                    from ..memory_storage import conversation_storage
                    rolling_summary = conversation_storage.get_conversation_summary(session_id, limit=8)
                    if rolling_summary:
                        workflow_state["rolling_summary"] = rolling_summary
                        # Extract context from fallback summary too
                        workflow_state["agricultural_context"] = self._extract_agricultural_context(rolling_summary)
                        logger.debug("Using fallback conversation history")
                except Exception as _fallback_err:
                    logger.debug("Fallback history also not available: %s", str(_fallback_err))
            
            # Check if this is a conversation history question and handle it directly
            query = user_input.get("query", "").lower()
            if any(phrase in query for phrase in ["what was my last question", "what did i ask", "previous question", "last question"]):
                rolling_summary = workflow_state.get("rolling_summary", "")
                if rolling_summary:
                    # Extract the most recent farmer question from the summary
                    farmer_questions = []
                    parts = rolling_summary.split(" | ")
                    for part in parts:
                        if part.startswith("Farmer: "):
                            farmer_questions.append(part[8:])  # Remove "Farmer: " prefix
                    
                    if farmer_questions:
                        # Get the most recent farmer question (excluding the current one)
                        last_question = farmer_questions[-1] if len(farmer_questions) > 1 else farmer_questions[0]
                        
                        # Return a direct response about the last question
                        workflow_state["direct_response"] = f"Based on our conversation history, your last question was: \"{last_question}\"\n\nIs there anything specific about this topic you'd like to explore further?"
                        logger.info("Handled conversation history question directly")

            # Check if we have a direct response for conversation history questions
            if "direct_response" in workflow_state:
                logger.info("🎯 Returning direct response for conversation history question")
                return {
                    "workflow_id": workflow_id,
                    "session_id": session_id,
                    "workflow_type": workflow_type.value,
                    "status": "completed",
                    "start_time": workflow_state["start_time"],
                    "agents_executed": ["conversation_history"],
                    "results": {
                        "display_text": workflow_state["direct_response"],
                        "response": workflow_state["direct_response"]
                    },
                    "errors": [],
                    "metadata": {
                        "user_context": user_context,
                        "user_input": user_input
                    },
                    "rolling_summary": workflow_state.get("rolling_summary", ""),
                    "end_time": datetime.utcnow(),
                    "execution_time_seconds": (datetime.utcnow() - workflow_state["start_time"]).total_seconds()
                }
            
            # Ensure components are initialized
            await self._ensure_initialized()
            
            logger.info(f"🎯 Starting AI-orchestrated workflow for session {session_id}")
            
            if not self.coordinator_agent:
                # Fallback to basic orchestration if AI coordinator unavailable
                logger.warning("⚠️ AI Coordinator unavailable, using fallback orchestration")
                results = await self._execute_fallback_workflow(workflow_state, workflow_type)
            else:
                # Use AI coordinator for intelligent workflow execution
                logger.info("🤖 Using AI coordinator for workflow execution")
                results = await self._execute_ai_orchestrated_workflow(workflow_state)
            
            # Synthesize unified display text for UI
            try:
                # Enhance user_context with conversation history from workflow state
                enhanced_user_context = {**user_context}
                if "agricultural_context" in workflow_state:
                    enhanced_user_context["agricultural_context"] = workflow_state["agricultural_context"]
                if "rolling_summary" in workflow_state:
                    enhanced_user_context["rolling_summary"] = workflow_state["rolling_summary"]
                
                display_text = await self._synthesize_display_text(results, user_input, enhanced_user_context)
                if display_text:
                    results["display_text"] = display_text
            except Exception as _synth_err:
                logger.warning(f"⚠️ Failed to synthesize display text: {_synth_err}")

            # Update workflow state
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            workflow_state.update({
                "status": WorkflowStatus.COMPLETED.value,
                "end_time": datetime.utcnow(),
                "execution_time_seconds": execution_time,
                "results": results
            })
            
            # Update metrics
            self._update_metrics(workflow_state)
            
            logger.info(f"✅ Workflow {workflow_type.value} completed in {execution_time:.2f}s")
            
            return workflow_state
            
        except Exception as e:
            # Handle workflow failure
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            workflow_state.update({
                "status": WorkflowStatus.FAILED.value,
                "end_time": datetime.utcnow(),
                "execution_time_seconds": execution_time,
                "error": str(e)
            })
            
            logger.error(f"❌ Workflow {workflow_type.value} failed: {str(e)}")
            self._update_metrics(workflow_state)
            
            return workflow_state
        
        finally:
            # Cleanup workflow from active list
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

    async def _synthesize_display_text(self, results: Dict[str, Any], user_input: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """
        Create a single dynamic, user-facing response from perception + knowledge.
        Keeps internal metrics hidden; adapts to question intent.
        """
        try:
            query_text = (user_input.get("query") or "").strip()
            has_images = bool(user_input.get("images"))

            knowledge = (results or {}).get("knowledge") or {}
            perception = (results or {}).get("perception") or {}
            image_analysis = perception.get("image_analysis") or {}

            # 1) Prefer knowledge agent's enhanced response first (works for both image and text queries)
            #    Knowledge agent receives perception context and can provide enriched responses
            for key in ["display_text", "contextual_advice", "content", "answer", "summary"]:
                text = knowledge.get(key)
                if isinstance(text, str) and text.strip() and not any(error_word in text.lower() for error_word in ["unavailable", "error", "failed", "temporarily"]):
                    body = self._sanitize_display_text(text.strip())
                    opener = self._generate_conversational_opener(user_input, user_context)
                    return (opener + body) if opener else body

            # 2) Fallback to perception's analysis_text if knowledge agent didn't provide good results
            #    This ensures we still show the image diagnosis when knowledge enhancement fails
            if has_images and (image_analysis or perception):
                analysis_text = (
                    image_analysis.get("analysis_text") or
                    perception.get("analysis_text")
                )
                if isinstance(analysis_text, str) and analysis_text.strip():
                    body = self._sanitize_display_text(analysis_text.strip())
                    opener = self._generate_conversational_opener(user_input, user_context)
                    return (opener + body) if opener else body

            # 3) Last resort: Let LLM handle the synthesis even without structured data
            #    Using it at the end prevents it from overwriting high-signal perception outputs.
            synthesized = await self._synthesize_with_llm(results, user_input, user_context)
            body = self._sanitize_display_text(synthesized or "I've analyzed your request and I'm here to help with your farming questions. What would you like to know more about?")
            opener = self._generate_conversational_opener(user_input, user_context)
            return (opener + body) if opener else body
        except Exception:
            return "I've analyzed your request and prepared recommendations."

    def _sanitize_display_text(self, text: str) -> str:
        """Hide internal structured sections and keep only conversational analysis.

        - Prefer content after an ANALYSIS_TEXT header if present
        - Remove lines that look like internal keys (HEALTH_SCORE, CONFIDENCE, etc.)
        - Strip markdown headings that expose internal sections
        - Collapse multiple blank lines
        """
        if not isinstance(text, str):
            return ""

        value = text.strip()

        # First, try to extract just the ANALYSIS_TEXT portion
        analysis_match = re.search(r"ANALYSIS_TEXT\s*:?\s*\n(.*)$", value, flags=re.IGNORECASE | re.DOTALL)
        if analysis_match:
            # Found ANALYSIS_TEXT section, use only that
            return analysis_match.group(1).strip()

        # If no ANALYSIS_TEXT section, clean up the whole text
        lines = []
        skip_section = False
        
        for line in value.splitlines():
            line_stripped = line.strip()
            
            # Skip internal metadata lines
            if re.match(r"^\s*(HEALTH_SCORE|CONFIDENCE|ISSUES|SEVERITY|RECOMMENDATIONS|OBSERVATIONS|Urgency Level|Growth Stage|Disease Detection|Nutrient Status|Environmental Stress|Health Assessment|Visual Observations)\s*:", line, flags=re.IGNORECASE):
                skip_section = True
                continue
                
            # Skip markdown section headers that look internal
            if line_stripped.startswith("###") and any(keyword in line_stripped.lower() for keyword in ["health", "confidence", "severity", "recommendations", "observations", "urgency"]):
                skip_section = True
                continue
                
            # Reset skip if we hit another section or content
            if line_stripped and not line_stripped.startswith("###") and ":" not in line_stripped:
                skip_section = False
                
            if not skip_section and line_stripped:
                lines.append(line)

        cleaned = "\n".join(lines)
        
        # Collapse excessive blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        # Remove generic greetings at the very start (to keep continuity tone)
        cleaned = re.sub(
            r"^(?:\s*(?:hey there|hello|hi|hey|greetings|hi there)[!.,\s]*)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).lstrip()
        
        # If we still have structured content, try to find the last paragraph that looks conversational
        if any(keyword in cleaned.upper() for keyword in ["HEALTH_SCORE", "CONFIDENCE", "SEVERITY"]):
            # Split by double newlines and take the last substantial paragraph
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            for paragraph in reversed(paragraphs):
                # Look for paragraphs that don't contain structured keywords
                if not any(keyword in paragraph.upper() for keyword in ["HEALTH_SCORE", "CONFIDENCE", "SEVERITY", "ISSUES"]):
                    if len(paragraph) > 50:  # Substantial content
                        return paragraph
        
        return cleaned

    def _generate_conversational_opener(self, user_input: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Create a short, optional bridge sentence that feels contextual and non-repetitive.

        - Uses rolling summary and agricultural context when present
        - Skips if it's a fresh topic or adds no value
        - Never includes explicit greetings; returns with trailing space if used
        """
        try:
            query = (user_input.get("query") or "").strip().lower()
            if not query:
                return ""

            # Detect intent from the query
            is_follow_up = any(k in query for k in ["follow", "again", "update", "continue", "as we", "about that", "that issue"])  # heuristic
            is_new_topic = any(k in query for k in ["new topic", "another crop", "different crop"])  # heuristic

            rolling = (user_context or {}).get("rolling_summary") or ""
            ag_ctx = (user_context or {}).get("agricultural_context") or {}
            crops = list(ag_ctx.get("crops", []))
            problems = list(ag_ctx.get("problems", []))

            # If it's clearly a new topic, skip opener
            if is_new_topic:
                return ""

            # If we have no useful context, skip opener
            if not rolling and not crops and not problems and not is_follow_up:
                return ""

            # Build a concise opener fragment from available signals
            fragments = []
            if crops:
                crop_phrase = ", ".join(sorted(crops)[:2])
                fragments.append(f"about your {crop_phrase}")
            if problems:
                problem_map = {
                    "pest_infestation": "pest issue",
                    "plant_disease": "disease concern",
                    "nutrient_deficiency": "nutrient question",
                    "water_stress": "watering challenge",
                    "soil_issues": "soil condition",
                }
                readable = ", ".join(problem_map.get(p, p.replace("_", " ")) for p in sorted(problems)[:2])
                fragments.append(f"on the {readable}")

            # Candidate templates (no greetings; professional and brief)
            templates = [
                "Building on what we've covered {frag}, ",
                "Following up {frag}, ",
                "Regarding {frag}, ",
                "For continuity {frag}, ",
            ]

            # Assemble fragment
            frag_text = " ".join(fragments).strip()
            if not frag_text and not is_follow_up:
                return ""

            template = random.choice(templates)
            opener = template.format(frag=("" if is_follow_up and not frag_text else frag_text))
            opener = re.sub(r"\s+", " ", opener).strip()

            # Ensure opener ends with a space and does not exceed ~100 chars
            if len(opener) > 100:
                opener = opener[:97].rstrip() + "… "
            else:
                opener = opener + " " if not opener.endswith(" ") else opener
            return opener
        except Exception:
            return ""

    async def _synthesize_with_llm(self, results: Dict[str, Any], user_input: Dict[str, Any], user_context: Dict[str, Any]) -> Optional[str]:
        """
        Use a lightweight model to synthesize a final display_text from perception + knowledge.
        Includes instruction to use emojis sparingly for engagement.
        """
        try:
            if not settings.OPENAI_API_KEY:
                return None
            model = getattr(settings, "GENERAL_CHAT_MODEL", None) or settings.OPENAI_MODEL

            knowledge = (results or {}).get("knowledge") or {}
            perception = (results or {}).get("perception") or {}
            image_analysis = perception.get("image_analysis") or {}

            # Extract evidence
            detected_issues = image_analysis.get("detected_issues") or perception.get("detected_issues") or []
            observations = (image_analysis.get("metadata") or {}).get("observations") or ""
            severity = image_analysis.get("severity") or perception.get("severity")
            confidence = image_analysis.get("confidence_level") or perception.get("confidence_level")

            best_answer = (
                knowledge.get("display_text") or
                knowledge.get("contextual_advice") or
                knowledge.get("content") or ""
            )
            actions = []
            if isinstance(knowledge.get("relevant_documents"), list):
                # Not guaranteed to contain actions; leave empty unless extracted elsewhere
                actions = []

            # Intent heuristic
            user_q = (user_input.get("query") or "").lower()
            intent = "general"
            if any(k in user_q for k in ["what disease", "diagnos", "identify"]):
                intent = "diagnosis"
            if any(k in user_q for k in ["treat", "control", "manage", "fix"]):
                intent = "treatment" if intent != "diagnosis" else "diagnosis+treatment"

            prompt = self._build_synthesizer_prompt(
                user_question=user_input.get("query") or "",
                user_context=user_context or {},
                perception_evidence={
                    "detected_issues": detected_issues,
                    "observations": observations,
                    "severity": severity,
                    "confidence": confidence,
                },
                knowledge_evidence={
                    "best_answer": best_answer,
                    "actions": actions,
                    "rationale": "",
                },
                intent=intent,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.7,
            )

            chat = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are AgriPal, an agricultural assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=min(max(settings.OPENAI_TEMPERATURE, 0.2), 0.9),
                max_tokens=min(settings.OPENAI_MAX_TOKENS, 600),
            )
            text = chat.choices[0].message.content if chat and chat.choices else None
            if isinstance(text, str):
                return text.strip()
            return None
        except Exception as e:
            logger.warning("LLM synthesis failed: %s", str(e))
            return None

    def _build_synthesizer_prompt(
        self,
        user_question: str,
        user_context: Dict[str, Any],
        perception_evidence: Dict[str, Any],
        knowledge_evidence: Dict[str, Any],
        intent: str,
        confidence: float,
    ) -> str:
        """Builds the LLM prompt for creating natural, engaging, and strategic responses."""
        location = user_context.get("location") or "Unknown"
        crop = user_context.get("crop_type") or "Unknown"
        
        # Include conversation history context if available
        agricultural_context = user_context.get("agricultural_context", {})
        rolling_summary = user_context.get("rolling_summary", "")
        
        conversation_context = ""
        if rolling_summary:
            conversation_context = f"\nCONVERSATION HISTORY:\n{rolling_summary}\n"
        
        if agricultural_context:
            crops_discussed = ", ".join(agricultural_context.get("crops", [])) or "None mentioned"
            problems_discussed = ", ".join(agricultural_context.get("problems", [])) or "None mentioned"
            solutions_discussed = ", ".join(agricultural_context.get("solutions", [])) or "None mentioned"
            conversation_context += (
                f"\nFARMING CONTEXT FROM PREVIOUS CONVERSATIONS:\n"
                f"- Crops discussed: {crops_discussed}\n"
                f"- Problems discussed: {problems_discussed}\n"
                f"- Solutions discussed: {solutions_discussed}\n"
            )
        
        return (
            "You are AgriPal, an experienced agricultural consultant and trusted farming advisor. "
            "Your role is to provide strategic insights, meaningful guidance, and engaging conversations "
            "that help farmers make informed decisions. Be conversational, insightful, and genuinely helpful.\n\n"
            "RESPONSE GUIDELINES:\n"
            "- Write naturally, as if continuing an ongoing conversation with a farmer you know well\n"
            "- AVOID repetitive greetings like 'Hey there!' or 'Hello!' - you're already in conversation\n"
            "- Reference previous conversations naturally when relevant (e.g., 'Following up on the corn issue we discussed...', 'Building on what we covered earlier...')\n"
            "- Show continuity by acknowledging the farmer's ongoing journey and challenges\n"
            "- Build on previous context to demonstrate understanding and memory\n"
            "- Provide strategic insights that go beyond basic information\n"
            "- Be engaging and show genuine interest in their farming progress\n"
            "- Use evidence to support your insights, but don't expose technical details\n"
            "- Adapt your tone based on whether this is a new topic or continuation\n"
            "- Offer practical wisdom and real-world context when relevant\n"
            "- Ask thoughtful follow-up questions that show you remember their situation\n"
            "- Be encouraging and supportive, especially for ongoing challenges\n"
            "- Use relevant emojis throughout your response to make it more engaging and visually appealing\n"
            "- Include 3-5 emojis per response that relate to the content (🌾🌱🌿💧☀️🌧️🐛🔬📊✅❌⚠️💡🎯🚀)\n\n"
            f"FARMER'S QUESTION: {user_question}\n"
            f"FARM CONTEXT: {crop} in {location}\n"
            f"QUESTION TYPE: {intent}\n"
            f"{conversation_context}\n"
            "AVAILABLE EVIDENCE (use to inform your response, but don't list directly):\n"
            f"- Visual Analysis: {perception_evidence.get('detected_issues', [])}\n"
            f"- Observations: {perception_evidence.get('observations', '')}\n"
            f"- Severity Level: {perception_evidence.get('severity', 'unknown')}\n"
            f"- Knowledge Base: {knowledge_evidence.get('best_answer', '')}\n"
            f"- Recommended Actions: {knowledge_evidence.get('actions', [])}\n\n"
            "Create a response that:\n"
            "1. Directly addresses their question with strategic insight\n"
            "2. References relevant previous conversations naturally when appropriate\n"
            "3. Provides meaningful context and practical guidance\n"
            "4. Engages them in a natural conversation\n"
            "5. Offers actionable next steps when appropriate\n"
            "6. Shows understanding of their farming context and history\n\n"
            "Write your response now:\n"
            "- Optionally add brief context such as why this happens or what to monitor next.\n"
            "- If confidence is limited, be candid and suggest verification or alternatives.\n"
            "- Personalize to the crop and location when relevant.\n"
            "- End with exactly ONE tailored follow-up based on the user's intent and missing info.\n"
            "  Examples (choose one dynamically): ask for images/symptoms (diagnosis), inquire about available inputs or constraints (treatment),\n"
            "  ask about timing/rotation/history (prevention), or ask what would be most useful next (general). Do not suggest products unless the user asked.\n"
            "- Keep it scannable and within a few short paragraphs total.\n"
            "- DO NOT include generic disclaimers like 'consult local experts' or 'contact extension services' unless specifically relevant to the situation.\n"
            "- Focus on providing actionable, specific advice based on the evidence provided.\n"
            "- Do NOT show raw scores, severity labels, or tool output formatting.\n"
            "- Keep it friendly, clear, and farmer-focused, with 0–4 relevant emojis.\n"
        )
    
    async def _execute_ai_orchestrated_workflow(self, workflow_state: Dict) -> Dict[str, Any]:
        """
        Execute workflow using AI coordinator for intelligent decision making
        """
        session_id = workflow_state["session_id"]
        user_input = workflow_state["metadata"]["user_input"]
        user_context = workflow_state["metadata"]["user_context"]
        original_workflow_type = workflow_state.get("workflow_type", WorkflowType.FULL_CONSULTATION.value)
        
        # Build prompt for AI coordinator
        coordinator_prompt = self._build_coordinator_prompt(user_input, user_context, workflow_state)
        
        try:
            # Prefer SDK only if a compatible run method exists; otherwise fall back
            if self.coordinator_agent:
                run_method = getattr(self.coordinator_agent, "run", None)
                if callable(run_method):
                    run_result = await run_method(
                        message=coordinator_prompt,
                        tools_context={
                            "user_input": user_input,
                            "user_context": user_context,
                            "workflow_state": workflow_state
                        }
                    )
                    # Parse results from coordinator agent
                    results = await self._parse_coordinator_response(run_result, workflow_state)
                    return results
            
                logger.info("ℹ️ Coordinator agent does not support run(); using fallback orchestration")
                return await self._execute_fallback_workflow(
                    workflow_state, WorkflowType(original_workflow_type)
                )
            
        except Exception as e:
            logger.error(f"❌ AI coordination failed: {str(e)}")
            # Fallback to basic workflow execution
            return await self._execute_fallback_workflow(
                workflow_state, WorkflowType(original_workflow_type)
            )
    
    def _build_coordinator_prompt(
        self, 
        user_input: Dict[str, Any], 
        user_context: Dict[str, Any],
        workflow_state: Dict[str, Any]
    ) -> str:
        """
        Build prompt for AI coordinator agent
        """
        images_provided = bool(user_input.get("images"))
        email_requested = bool(user_input.get("recipient_email"))
        query = user_input.get("query", "")
        crop_type = user_input.get("crop_type", "unknown")
        # Include short rolling summary if available
        rolling_summary = (
            workflow_state.get("rolling_summary")
            or (user_context.get("rolling_summary") if isinstance(user_context, dict) else None)
            or ""
        )
        
        return f"""
        Please orchestrate an agricultural consultation workflow for this farmer request.
        
        Request Details:
        - Query: {query}
        - Crop Type: {crop_type}
        - Images Provided: {images_provided} ({len(user_input.get("images", []))} images)
        - Email Report Requested: {email_requested}
        - User Location: {user_context.get("location", "Not specified")}
        - Recent Context Summary: {rolling_summary}
        
        IMPORTANT: If the user is asking about previous questions or conversation history, use the "Recent Context Summary" above to provide accurate information about what was discussed previously. The context shows the conversation flow with "Farmer:" and "AgriPal:" labels.
        
        For questions like "what was my last question" or "what did I ask before", look at the Recent Context Summary and identify the most recent "Farmer:" entry to find the user's previous question. Always provide specific details from the conversation history when available.
        
        CRITICAL: When the user asks about their previous questions, DO NOT say you can't access them. The conversation history is provided above in the "Recent Context Summary". Use it to give a direct, specific answer about what the user asked previously.
        
        Available Agents:
        - Perception Agent: Analyze images for crop health, disease detection, etc.
        - Knowledge Agent: Search agricultural knowledge base and research data
        - Email Agent: Generate and send comprehensive consultation reports
        
        Your task:
        1. Analyze the user request to determine which agents are needed
        2. If the user is asking about previous questions or conversation history, use the Recent Context Summary to provide a direct answer WITHOUT executing any agents
        3. For conversation history questions, provide a direct response using the context summary
        4. Plan the optimal execution order (sequential/parallel) for other requests
        5. Execute the agents with proper data passing
        6. Handle any errors with appropriate fallback strategies
        7. Optimize for best user experience
        
        Use the available tools to execute this workflow intelligently.
        """
    
    async def _parse_coordinator_response(self, run_result: RunResult, workflow_state: Dict) -> Dict[str, Any]:
        """
        Parse response from AI coordinator agent
        """
        results = {}
        
        # Extract tool execution results
        if hasattr(run_result, 'tool_calls') and run_result.tool_calls:
            for tool_call in run_result.tool_calls:
                if tool_call.function.name == "execute_perception_agent":
                    results["perception"] = json.loads(tool_call.function.result) if tool_call.function.result else None
                    workflow_state["agents_executed"].append("perception")
                    
                elif tool_call.function.name == "execute_knowledge_agent":
                    results["knowledge"] = json.loads(tool_call.function.result) if tool_call.function.result else None
                    workflow_state["agents_executed"].append("knowledge")
                    
                elif tool_call.function.name == "execute_email_agent":
                    results["email"] = json.loads(tool_call.function.result) if tool_call.function.result else None
                    workflow_state["agents_executed"].append("email")
        
        # Also extract the coordinator's own assistant message (what you see in SDK traces)
        try:
            final_text = None
            # Common SDK patterns: output, output_text, messages[-1]
            final_text = getattr(run_result, "output_text", None) or getattr(run_result, "output", None)
            if not final_text and hasattr(run_result, "messages") and run_result.messages:
                last_msg = run_result.messages[-1]
                # messages may be dicts or objects with role/content
                content = None
                if isinstance(last_msg, dict):
                    content = last_msg.get("content")
                else:
                    content = getattr(last_msg, "content", None)
                if isinstance(content, list):
                    # Handle content blocks
                    text_blocks = [b.get("text") if isinstance(b, dict) else str(b) for b in content]
                    final_text = "\n".join([t for t in text_blocks if t])
                elif isinstance(content, str):
                    final_text = content
            if isinstance(final_text, str) and final_text.strip():
                results["display_text"] = final_text.strip()
        except Exception as e:
            logger.warning("Failed to extract coordinator final message: %s", str(e))

        return results
    
    async def _execute_fallback_workflow(self, workflow_state: Dict, workflow_type: WorkflowType) -> Dict[str, Any]:
        """
        Fallback workflow execution when AI coordinator is unavailable
        """
        session_id = workflow_state["session_id"]
        user_input = workflow_state["metadata"]["user_input"]
        user_context = workflow_state["metadata"]["user_context"]
        
        results = {}
        # Ensure session cache entry
        session_cache_entry = self.session_cache.get(session_id) or {}
        self.session_cache[session_id] = session_cache_entry
        
        # Basic workflow logic
        logger.info(f"🔍 Workflow type: {workflow_type}, Has images: {bool(user_input.get('images'))}")
        logger.info(f"🔍 Workflow type enum: {WorkflowType.PERCEPTION_TO_KNOWLEDGE}")
        logger.info(f"🔍 Images in user_input: {user_input.get('images')}")
        
        # Handle different workflow types
        logger.info(f"🔍 Checking workflow type: {workflow_type} == {WorkflowType.PERCEPTION_TO_KNOWLEDGE}")
        logger.info(f"🔍 Workflow type comparison result: {workflow_type == WorkflowType.PERCEPTION_TO_KNOWLEDGE}")
        
        if workflow_type == WorkflowType.PERCEPTION_TO_KNOWLEDGE:
            # Specific workflow: Perception → Knowledge
            logger.info("🎯 Executing PERCEPTION_TO_KNOWLEDGE workflow")
            if user_input.get("images"):
                logger.info(f"📸 Running perception-to-knowledge workflow with {len(user_input.get('images', []))} images")
                perception_results = await self._run_perception_agent(session_id, user_input, user_context)
                if perception_results:
                    logger.info("✅ Perception agent completed successfully")
                    results["perception"] = perception_results
                    workflow_state["agents_executed"].append("perception")
                    # Persist latest perception for this session
                    session_cache_entry["last_perception"] = perception_results
                else:
                    logger.warning("⚠️ Perception agent returned no results")
            else:
                logger.warning("⚠️ PERCEPTION_TO_KNOWLEDGE workflow requires images but none provided")
                
        elif workflow_type == WorkflowType.FULL_CONSULTATION or user_input.get("images"):
            # Run perception if images provided
            if user_input.get("images"):
                logger.info(f"📸 Running perception agent with {len(user_input.get('images', []))} images")
                perception_results = await self._run_perception_agent(session_id, user_input, user_context)
                if perception_results:
                    logger.info("✅ Perception agent completed successfully")
                    results["perception"] = perception_results
                    workflow_state["agents_executed"].append("perception")
                    # Persist latest perception for this session
                    session_cache_entry["last_perception"] = perception_results
                else:
                    logger.warning("⚠️ Perception agent returned no results")
            else:
                logger.info("📝 No images provided, skipping perception agent")
        else:
            logger.info("📝 Workflow type does not require perception agent")

        # General chat intent detection vs knowledge search
        query = (user_input.get("query") or "").strip()
        if query:
            # For PERCEPTION_TO_KNOWLEDGE workflow, always run knowledge agent
            if workflow_type == WorkflowType.PERCEPTION_TO_KNOWLEDGE:
                logger.info("🔍 PERCEPTION_TO_KNOWLEDGE workflow: Running knowledge agent with perception context")
                # Determine perception context to pass
                perception_context_to_pass = results.get("perception")
                # Optional flags from user_input
                clear_flag = bool(user_input.get("clear_perception"))
                reuse_flag = user_input.get("reuse_last_perception")
                if clear_flag:
                    session_cache_entry.pop("last_perception", None)
                if perception_context_to_pass is None:
                    # Default behavior: reuse if available unless explicitly disabled
                    should_reuse = True if reuse_flag is None else bool(reuse_flag)
                    if should_reuse:
                        perception_context_to_pass = session_cache_entry.get("last_perception")

                # Enhance user_context with conversation history from workflow state
                enhanced_user_context = {**user_context}
                if "agricultural_context" in workflow_state:
                    enhanced_user_context["agricultural_context"] = workflow_state["agricultural_context"]
                if "rolling_summary" in workflow_state:
                    enhanced_user_context["rolling_summary"] = workflow_state["rolling_summary"]

                knowledge_results = await self._run_knowledge_agent(
                    session_id, user_input, enhanced_user_context, perception_context_to_pass
                )
                if knowledge_results:
                    results["knowledge"] = knowledge_results
                    workflow_state["agents_executed"].append("knowledge")
            elif self._is_general_chat_intent(query):
                # Use a cheaper model for general conversation (opportunistically enriched)
                general_response = await self._generate_general_chat_response(
                    query, user_context, session_id, user_input.get("crop_type")
                )
                if general_response:
                    results["response"] = general_response
                    workflow_state["agents_executed"].append("general_chat")
            else:
                # Run knowledge agent for agricultural/informational queries
                # Determine perception context to pass
                perception_context_to_pass = results.get("perception")
                # Optional flags from user_input
                clear_flag = bool(user_input.get("clear_perception"))
                reuse_flag = user_input.get("reuse_last_perception")
                if clear_flag:
                    session_cache_entry.pop("last_perception", None)
                if perception_context_to_pass is None:
                    # Default behavior: reuse if available unless explicitly disabled
                    should_reuse = True if reuse_flag is None else bool(reuse_flag)
                    if should_reuse:
                        perception_context_to_pass = session_cache_entry.get("last_perception")

                # Enhance user_context with conversation history from workflow state
                enhanced_user_context = {**user_context}
                if "agricultural_context" in workflow_state:
                    enhanced_user_context["agricultural_context"] = workflow_state["agricultural_context"]
                if "rolling_summary" in workflow_state:
                    enhanced_user_context["rolling_summary"] = workflow_state["rolling_summary"]

                knowledge_results = await self._run_knowledge_agent(
                    session_id, user_input, enhanced_user_context, perception_context_to_pass
                )
                if knowledge_results:
                    results["knowledge"] = knowledge_results
                    workflow_state["agents_executed"].append("knowledge")
        
        # Run email agent if email provided
        if user_input.get("recipient_email"):
            email_results = await self._run_email_agent(
                session_id, user_input, user_context, 
                results.get("perception"), results.get("knowledge")
            )
            if email_results:
                results["email"] = email_results
                workflow_state["agents_executed"].append("email")
        
        return results

    def _is_general_chat_intent(self, query: str) -> bool:
        """Heuristic to detect small-talk/general chat vs domain questions.
        Returns True for general chat.
        """
        text = query.lower()
        # If it contains clear agri/tech terms, treat as knowledge
        domain_keywords = [
            "crop", "soil", "irrigation", "pest", "disease", "harvest", "fertilizer",
            "fungicide", "insecticide", "yield", "planting", "germination", "weed",
            "root", "leaf", "tomato", "wheat", "corn", "rice", "soybean", "cotton",
            "nitrogen", "phosphorus", "potassium", "agri", "farm", "field", "blight",
            "rust", "mildew", "aphid", "thrips", "bollworm", "stem borer"
        ]
        if any(k in text for k in domain_keywords):
            return False
        # Obvious chit-chat cues
        chit_chat = ["hi", "hello", "hey", "how are you", "who are you", "tell me a joke", "thanks", "thank you"]
        if any(k in text for k in chit_chat):
            return True
        # Generic questions without domain cues are treated as general chat
        return True

    async def _generate_general_chat_response(self, query: str, user_context: Dict[str, Any], session_id: str, crop_type: Optional[str] = None) -> Optional[str]:
        """Generate a concise, friendly response using a lower-cost model, with optional KB/weather enrichment."""
        try:
            model = getattr(settings, "GENERAL_CHAT_MODEL", None) or settings.OPENAI_MODEL
            # Lightweight system prompt with empathetic tone
            system_prompt = (
                "You are AgriPal, a friendly, concise assistant. Be helpful and upbeat. "
                "Use 1–2 tasteful emojis when it enhances clarity or warmth (avoid overuse). "
                "If relevant, relate answers to agriculture briefly, but avoid fabricating facts."
            )
            # Opportunistic enrichment from knowledge agent (includes weather when location present)
            enrichment_snippet = None
            try:
                kb_input = {"query": query, "crop_type": crop_type}
                kb_results = await self._run_knowledge_agent(session_id, kb_input, user_context, None)
                if kb_results:
                    parts: List[str] = []
                    advice = kb_results.get("contextual_advice")
                    if advice:
                        parts.append(advice[:500])
                    docs = kb_results.get("relevant_documents") or []
                    for d in docs[:2]:
                        content = (d.get("content") or d.get("page_content") or "")
                        if content:
                            parts.append(content[:200])
                    weather_block = kb_results.get("weather") or (kb_results.get("extras", {}) if isinstance(kb_results.get("extras"), dict) else {})
                    if isinstance(weather_block, dict) and weather_block.get("combined_recommendations"):
                        recs = weather_block.get("combined_recommendations")[:2]
                        if recs:
                            parts.append("Weather considerations: " + " | ".join([str(r) for r in recs]))
                    if parts:
                        enrichment_snippet = "\n".join(parts[:4])
            except Exception as e:
                logger.warning("KB enrichment skipped: %s", str(e))

            # Use Chat Completions API
            messages = [{"role": "system", "content": system_prompt}]
            if enrichment_snippet:
                messages.append({"role": "system", "content": "Optional context (may be partial):\n" + enrichment_snippet})
            messages.append({"role": "user", "content": query})
            completion = await self.client.chat.completions.create(
                model=model,
                temperature=0.6,
                messages=messages,
                max_tokens=300
            )
            return completion.choices[0].message.content if completion and completion.choices else None
        except Exception as e:
            logger.error("❌ General chat generation failed: %s", str(e))
            return None
    
    # Tool implementations for AI Coordinator Agent
    async def _tool_analyze_user_request(self, user_input: Dict, user_context: Dict = None) -> str:
        """
        Tool for analyzing user request to determine optimal workflow
        """
        try:
            analysis = {
                "needs_perception": bool(user_input.get("images")),
                "needs_knowledge": bool(user_input.get("query")),
                "needs_email": bool(user_input.get("recipient_email")),
                "crop_type": user_input.get("crop_type"),
                "complexity": "high" if len(user_input.get("images", [])) > 1 else "medium",
                "recommended_sequence": []
            }
            
            # Determine optimal sequence
            if analysis["needs_perception"]:
                analysis["recommended_sequence"].append("perception")
            if analysis["needs_knowledge"]:
                analysis["recommended_sequence"].append("knowledge")
            if analysis["needs_email"]:
                analysis["recommended_sequence"].append("email")
                
            return json.dumps(analysis)
            
        except Exception as e:
            logger.error(f"❌ User request analysis failed: {str(e)}")
            return json.dumps({"error": str(e), "recommended_sequence": ["knowledge"]})
    
    async def _tool_execute_perception_agent(self, images: List, analysis_type: str = "comprehensive", crop_type: str = None) -> str:
        """
        Tool for executing perception agent
        """
        try:
            if not self.perception_agent:
                return json.dumps({"error": "Perception agent not available", "results": None})
            
            # Mock session data for tool execution
            session_id = f"coord_{datetime.utcnow().timestamp()}"
            user_input = {"images": images, "analysis_type": analysis_type, "crop_type": crop_type}
            
            results = await self._run_perception_agent(session_id, user_input, {})
            return json.dumps({"success": True, "results": results})
            
        except Exception as e:
            logger.error(f"❌ Perception agent execution failed: {str(e)}")
            return json.dumps({"error": str(e), "results": None})
    
    async def _tool_execute_knowledge_agent(self, query: str, crop_type: str = None, perception_results: Dict = None) -> str:
        """
        Tool for executing knowledge agent
        """
        try:
            if not self.knowledge_agent:
                return json.dumps({"error": "Knowledge agent not available", "results": None})
            
            # Mock session data for tool execution
            session_id = f"coord_{datetime.utcnow().timestamp()}"
            user_input = {"query": query, "crop_type": crop_type}
            
            results = await self._run_knowledge_agent(session_id, user_input, {}, perception_results)
            return json.dumps({"success": True, "results": results})
            
        except Exception as e:
            logger.error(f"❌ Knowledge agent execution failed: {str(e)}")
            return json.dumps({"error": str(e), "results": None})
    
    async def _tool_execute_email_agent(self, recipient_email: str, session_data: Dict, analysis_results: Dict = None) -> str:
        """
        Tool for executing email agent
        """
        try:
            if not self.email_agent:
                return json.dumps({"error": "Email agent not available", "results": None})
            
            # Mock session data for tool execution
            session_id = f"coord_{datetime.utcnow().timestamp()}"
            user_input = {"recipient_email": recipient_email, "session_data": session_data}
            
            results = await self._run_email_agent(session_id, user_input, {}, 
                                               analysis_results.get("perception") if analysis_results else None,
                                               analysis_results.get("knowledge") if analysis_results else None)
            return json.dumps({"success": True, "results": results})
            
        except Exception as e:
            logger.error(f"❌ Email agent execution failed: {str(e)}")
            return json.dumps({"error": str(e), "results": None})
    
    async def _tool_handle_agent_error(self, agent_name: str, error_message: str, context: Dict = None) -> str:
        """
        Tool for handling agent errors with fallback strategies
        """
        try:
            fallback_strategy = {
                "agent": agent_name,
                "error": error_message,
                "fallback_action": "retry",
                "max_retries": 2,
                "alternative_approach": None
            }
            
            # Define fallback strategies per agent
            if agent_name == "perception":
                fallback_strategy["alternative_approach"] = "proceed_without_image_analysis"
            elif agent_name == "knowledge":
                fallback_strategy["alternative_approach"] = "use_cached_knowledge"
            elif agent_name == "email":
                fallback_strategy["alternative_approach"] = "generate_basic_summary"
                
            logger.warning(f"⚠️ Handling {agent_name} error: {error_message}")
            return json.dumps(fallback_strategy)
            
        except Exception as e:
            logger.error(f"❌ Error handling failed: {str(e)}")
            return json.dumps({"error": str(e), "fallback_action": "abort"})
    
    async def _tool_optimize_workflow(self, workflow_state: Dict, agent_health: Dict = None, performance_metrics: Dict = None) -> str:
        """
        Tool for optimizing workflow execution
        """
        try:
            optimization = {
                "current_performance": "normal",
                "recommended_optimizations": [],
                "parallel_execution_possible": False,
                "estimated_time_savings": 0
            }
            
            # Analyze current workflow state
            agents_executed = workflow_state.get("agents_executed", [])
            
            # Check if parallel execution is beneficial
            if agent_health and all(agent_health.values()):
                optimization["parallel_execution_possible"] = True
                optimization["estimated_time_savings"] = 30  # 30% time savings
                optimization["recommended_optimizations"].append("enable_parallel_execution")
            
            # Check for performance improvements
            if performance_metrics:
                avg_time = performance_metrics.get("average_execution_time", 0)
                if avg_time > 10:  # If average time > 10 seconds
                    optimization["recommended_optimizations"].append("enable_result_caching")
                    
            return json.dumps(optimization)
            
        except Exception as e:
            logger.error(f"❌ Workflow optimization failed: {str(e)}")
            return json.dumps({"error": str(e), "recommended_optimizations": []})
    
    async def _run_perception_agent(
        self, 
        session_id: str, 
        user_input: Dict, 
        user_context: Dict
    ) -> Optional[Dict]:
        """Run perception agent with error handling"""
        logger.info(f"🔍 _run_perception_agent called for session {session_id}")
        if not self.perception_agent:
            logger.warning("⚠️ Perception agent not available")
            return None
        
        try:
            # Extract image data from possible list inputs
            images_data = []
            images = user_input.get("images")
            if images:
                for item in images:
                    if isinstance(item, dict) and "content" in item:
                        images_data.append(item["content"])
                    elif isinstance(item, (bytes, bytearray)):
                        images_data.append(bytes(item))
            
            # Fallback to any direct 'image' key if provided
            if not images_data:
                direct_image = user_input.get("image")
                if direct_image:
                    images_data.append(direct_image)
            
            if not images_data:
                logger.warning("⚠️ No image data found in user_input")
                return None
            
            # For now, process the first image (can be extended to handle multiple images)
            image_bytes = images_data[0]
            logger.info(f"📸 Processing image: {len(image_bytes)} bytes")
            
            message = AgentMessage(
                session_id=session_id,
                agent_type=AgentType.PERCEPTION,
                content={
                    "image": image_bytes,
                    "query": user_input.get("query", ""),
                    "user_context": user_context,
                    "crop_type": user_input.get("crop_type"),
                    "analysis_type": user_input.get("analysis_type", "comprehensive")
                }
            )
            
            response = await self.perception_agent.process(message)
            return response.result if response.success else None
            
        except Exception as e:
            logger.error(f"❌ Perception agent failed: {str(e)}")
            return None
    
    async def _run_knowledge_agent(
        self, 
        session_id: str, 
        user_input: Dict, 
        user_context: Dict,
        perception_results: Dict = None
    ) -> Optional[Dict]:
        """Run knowledge agent with error handling"""
        if not self.knowledge_agent:
            logger.warning("⚠️ Knowledge agent not available")
            return None
        
        try:
            message = AgentMessage(
                session_id=session_id,
                agent_type=AgentType.KNOWLEDGE,
                content={
                    "query": user_input.get("query", ""),
                    "crop_type": user_input.get("crop_type"),
                    "user_context": user_context,
                    "perception_context": perception_results
                }
            )
            
            response = await self.knowledge_agent.process(message)
            return response.result if response.success else None
            
        except Exception as e:
            logger.error(f"❌ Knowledge agent failed: {str(e)}")
            return None
    
    async def _run_email_agent(
        self, 
        session_id: str, 
        user_input: Dict, 
        user_context: Dict,
        perception_results: Dict = None,
        knowledge_results: Dict = None
    ) -> Optional[Dict]:
        """Run email agent with error handling"""
        if not self.email_agent:
            logger.warning("⚠️ Email agent not available")
            return None
        
        try:
            # Prepare session data for email report
            session_data = {
                "messages": user_input.get("session_messages", []),
                "analysis_results": {
                    "perception": perception_results,
                    "knowledge": knowledge_results
                },
                "images_count": len(user_input.get("images", []))
            }
            
            message = AgentMessage(
                session_id=session_id,
                agent_type=AgentType.EMAIL,
                content={
                    "session_data": session_data,
                    "recipient_email": user_input.get("recipient_email"),
                    "user_context": user_context
                }
            )
            
            response = await self.email_agent.process(message)
            return response.result if response.success else None
            
        except Exception as e:
            logger.error(f"❌ Email agent failed: {str(e)}")
            return None
    
    def _update_metrics(self, workflow_state: Dict):
        """Update performance metrics"""
        self.workflow_metrics["total_workflows"] += 1
        
        if workflow_state["status"] == WorkflowStatus.COMPLETED.value:
            self.workflow_metrics["successful_workflows"] += 1
        else:
            self.workflow_metrics["failed_workflows"] += 1
        
        # Update average execution time
        if "execution_time_seconds" in workflow_state:
            current_avg = self.workflow_metrics["average_execution_time"]
            total_workflows = self.workflow_metrics["total_workflows"]
            new_time = workflow_state["execution_time_seconds"]
            
            self.workflow_metrics["average_execution_time"] = (
                (current_avg * (total_workflows - 1) + new_time) / total_workflows
            )
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get status of active workflow"""
        return self.active_workflows.get(workflow_id)
    
    async def get_active_workflows(self) -> List[Dict]:
        """Get all active workflows"""
        return list(self.active_workflows.values())
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get coordinator performance metrics"""
        agent_health = await self.health_check()
        
        return {
            "workflow_metrics": self.workflow_metrics,
            "agent_health": agent_health,
            "active_workflows_count": len(self.active_workflows),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all agents"""
        # Ensure components are initialized
        await self._ensure_initialized()
        
        health_checks = {}
        
        # Check each agent's health
        if self.perception_agent:
            try:
                perception_health = await self.perception_agent.health_check()
                health_checks["perception_agent"] = all(perception_health.values())
            except:
                health_checks["perception_agent"] = False
        else:
            health_checks["perception_agent"] = False
        
        if self.knowledge_agent:
            try:
                knowledge_health = await self.knowledge_agent.health_check()
                health_checks["knowledge_agent"] = all(knowledge_health.values())
            except:
                health_checks["knowledge_agent"] = False
        else:
            health_checks["knowledge_agent"] = False
        
        if self.email_agent:
            try:
                email_health = await self.email_agent.health_check()
                health_checks["email_agent"] = all(email_health.values())
            except:
                health_checks["email_agent"] = False
        else:
            health_checks["email_agent"] = False
        
        # Overall coordinator health
        health_checks["coordinator_healthy"] = any(health_checks.values())
        
        return health_checks
    
    def _extract_agricultural_context(self, content: str) -> Dict[str, Any]:
        """Extract agricultural context from conversation content"""
        context = {"crops": set(), "problems": set(), "solutions": set(), "location": ""}
        content_lower = content.lower()
        
        # Extract crops
        crop_keywords = {
            "corn": ["corn", "maize"],
            "rice": ["rice", "paddy"],
            "cassava": ["cassava", "tapioca"],
            "yam": ["yam"],
            "beans": ["beans", "cowpea"],
            "tomato": ["tomato", "tomatoes"],
            "pepper": ["pepper", "peppers"],
            "okra": ["okra"],
            "plantain": ["plantain"],
            "cocoa": ["cocoa", "cacao"],
            "coffee": ["coffee"]
        }
        
        for crop, keywords in crop_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                context["crops"].add(crop)
        
        # Extract common problems
        problem_keywords = {
            "pest_infestation": ["pest", "insect", "bug", "aphid", "caterpillar"],
            "plant_disease": ["disease", "fungus", "blight", "rot", "mold"],
            "nutrient_deficiency": ["yellow", "yellowing", "nutrient", "fertilizer"],
            "water_stress": ["drought", "wilting", "water"],
            "soil_issues": ["soil", "ph", "acidic"]
        }
        
        for problem, keywords in problem_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                context["problems"].add(problem)
        
        # Extract solutions mentioned
        solution_keywords = {
            "fertilization": ["fertilizer", "fertilize", "nutrient"],
            "pest_control": ["spray", "pesticide", "neem"],
            "irrigation": ["water", "irrigate", "irrigation"],
            "soil_treatment": ["lime", "compost", "manure"]
        }
        
        for solution, keywords in solution_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                context["solutions"].add(solution)
        
        # Extract location (basic)
        location_keywords = ["port harcourt", "lagos", "abuja", "kano", "ibadan", "nigeria"]
        for location in location_keywords:
            if location in content_lower:
                context["location"] = location
                break
        
        return context
    
    async def cleanup(self):
        """Cleanup coordinator resources"""
        # Cleanup all agents
        cleanup_tasks = []
        
        if self.perception_agent:
            cleanup_tasks.append(self.perception_agent.cleanup())
        if self.knowledge_agent:
            cleanup_tasks.append(self.knowledge_agent.cleanup())
        if self.email_agent:
            cleanup_tasks.append(self.email_agent.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear workflow state
        self.active_workflows.clear()
        self.session_cache.clear()
        
        logger.info("🧹 Agent Coordinator cleanup completed")
