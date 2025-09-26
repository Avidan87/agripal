"""
📧 Email Agent - Report Generation and Distribution
Specialized AI agent for generating and sending agricultural consultation reports
using OpenAI Agents SDK for intelligent report composition.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# Heavy imports - will be imported when needed
Template = None
sendgrid = None
Mail = None

# OpenAI Agents SDK imports
from openai import AsyncOpenAI
from agents import Agent, Tool

from ..models import (
    AgentMessage, 
    AgentResponse, 
    AgentType, 
    EmailSendResult
)
from ..config import settings

logger = logging.getLogger(__name__)

def _import_heavy_dependencies():
    """Import heavy dependencies only when needed"""
    global Template, sendgrid, Mail
    try:
        from jinja2 import Template
        import sendgrid
        from sendgrid.helpers.mail import Mail
        return True
    except ImportError as e:
        logger.warning(f"⚠️ Heavy dependencies not available: {e}")
        return False

class EmailAgent:
    """
    📧 Intelligent email report generation and distribution agent using OpenAI Agents SDK
    
    Capabilities:
    - Session summary generation using AI
    - Professional agricultural report formatting
    - Email delivery via SendGrid
    - Template-based report generation
    - Attachment handling for images and documents
    - Follow-up scheduling and recommendations
    """
    
    def __init__(self):
        self.agent_type = AgentType.EMAIL
        self.model = settings.OPENAI_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Check if heavy dependencies are available
        self.dependencies_available = _import_heavy_dependencies()
        
        # Email configuration - SendGrid only
        self.from_email = settings.SENDGRID_FROM_EMAIL
        self.from_name = settings.SENDGRID_FROM_NAME
        
        # Initialize SendGrid client
        self.sendgrid_client = None
        
        # Report templates
        self.templates = {}
        
        # Initialize OpenAI Agent SDK
        self.agent = None
        asyncio.create_task(self._initialize_async_components())
        
        logger.info("📧 Email Agent initialized with OpenAI Agents SDK")
    
    async def _initialize_async_components(self):
        """
        Initialize async components (email clients and agent)
        """
        try:
            # Setup email clients
            await self._setup_email_clients()
            
            # Load email templates
            await self._load_templates()
            
            # Setup OpenAI Agent with tools
            await self._setup_agent()
            
            logger.info("✅ Email Agent async components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Email Agent async components: {str(e)}")
    
    async def _setup_email_clients(self):
        """
        Initialize SendGrid email client
        """
        try:
            if not settings.SENDGRID_API_KEY:
                raise ValueError("SENDGRID_API_KEY is required but not provided")
            
            self.sendgrid_client = sendgrid.SendGridAPIClient(api_key=settings.SENDGRID_API_KEY)
            logger.info("✅ SendGrid client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup SendGrid client: {str(e)}")
            raise
    
    async def _load_templates(self):
        """
        Load email templates for different report types
        """
        try:
            # Session report template
            self.templates["session_report"] = Template("""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>AgriPal Consultation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .header { background-color: #2E7D32; color: white; padding: 20px; text-align: center; }
                    .content { padding: 20px; }
                    .section { margin-bottom: 30px; }
                    .finding { background-color: #E8F5E8; padding: 15px; border-radius: 5px; margin: 10px 0; }
                    .recommendation { background-color: #FFF3E0; padding: 15px; border-radius: 5px; margin: 10px 0; }
                    .footer { background-color: #f4f4f4; padding: 20px; text-align: center; font-size: 12px; }
                    .urgency-high { border-left: 5px solid #F44336; }
                    .urgency-medium { border-left: 5px solid #FF9800; }
                    .urgency-low { border-left: 5px solid #4CAF50; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🌾 AgriPal Consultation Report</h1>
                    <p>Agricultural Analysis & Recommendations</p>
                </div>
                
                <div class="content">
                    <div class="section">
                        <h2>Farmer Information</h2>
                        <p><strong>Name:</strong> {{ user_name or 'Farmer' }}</p>
                        <p><strong>Email:</strong> {{ user_email }}</p>
                        <p><strong>Location:</strong> {{ location or 'Not specified' }}</p>
                        <p><strong>Report Date:</strong> {{ report_date }}</p>
                        <p><strong>Session ID:</strong> {{ session_id }}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Session Summary</h2>
                        <p>{{ session_summary }}</p>
                    </div>
                    
                    {% if key_findings %}
                    <div class="section">
                        <h2>Key Findings</h2>
                        {% for finding in key_findings %}
                        <div class="finding">
                            <p>{{ finding }}</p>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    
                    {% if recommendations %}
                    <div class="section">
                        <h2>Recommendations</h2>
                        {% for recommendation in recommendations %}
                        <div class="recommendation">
                            <p>{{ recommendation }}</p>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    
                    {% if weather_data %}
                    <div class="section">
                        <h2>Weather Considerations</h2>
                        <p><strong>Current Conditions:</strong> {{ weather_data.current_temperature }}°C, {{ weather_data.humidity }}% humidity</p>
                        <p><strong>Precipitation:</strong> {{ weather_data.precipitation }}mm</p>
                        {% if weather_data.recommendations %}
                        <ul>
                        {% for rec in weather_data.recommendations %}
                            <li>{{ rec }}</li>
                        {% endfor %}
                        </ul>
                        {% endif %}
                    </div>
                    {% endif %}
                    
                    {% if images_analyzed > 0 %}
                    <div class="section">
                        <h2>Image Analysis</h2>
                        <p>{{ images_analyzed }} field image(s) were analyzed during this consultation.</p>
                    </div>
                    {% endif %}
                    
                    <div class="section">
                        <h2>Next Steps</h2>
                        <ul>
                            <li>Implement the recommended practices gradually</li>
                            <li>Monitor crop response to any interventions</li>
                            <li>Keep detailed records for future consultations</li>
                            <li>Contact local agricultural extension services if needed</li>
                            <li>Schedule follow-up consultation if conditions change</li>
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This report was generated by AgriPal AI Agricultural Assistant</p>
                    <p>For questions or follow-up consultations, please contact support@agripal.com</p>
                    <p><em>Disclaimer: This report provides general agricultural guidance. Always consult with local experts for specific conditions.</em></p>
                </div>
            </body>
            </html>
            """)
            
            logger.info("✅ Email templates loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load email templates: {str(e)}")
    
    async def _setup_agent(self):
        """
        Initialize the OpenAI Agent with email and report generation tools
        """
        try:
            # Define tools for the email agent
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_session_summary",
                        "description": "Generate an intelligent summary of the agricultural consultation session",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "session_messages": {
                                    "type": "array",
                                    "description": "List of session messages to summarize"
                                },
                                "analysis_results": {
                                    "type": "object",
                                    "description": "Results from perception and knowledge agents"
                                },
                                "user_context": {
                                    "type": "object",
                                    "description": "User profile and context information"
                                }
                            },
                            "required": ["session_messages"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "extract_key_findings",
                        "description": "Extract and prioritize key findings from the consultation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "session_data": {
                                    "type": "object",
                                    "description": "Complete session data including analysis results"
                                },
                                "perception_results": {
                                    "type": "object",
                                    "description": "Visual analysis results from perception agent"
                                },
                                "knowledge_results": {
                                    "type": "object",
                                    "description": "Knowledge search and recommendations"
                                }
                            },
                            "required": ["session_data"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "format_recommendations",
                        "description": "Format and prioritize actionable recommendations for the farmer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "raw_recommendations": {
                                    "type": "array",
                                    "description": "Raw recommendations from various agents"
                                },
                                "urgency_levels": {
                                    "type": "array",
                                    "description": "Urgency levels for each recommendation"
                                },
                                "user_context": {
                                    "type": "object",
                                    "description": "User context for personalization"
                                }
                            },
                            "required": ["raw_recommendations"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email_report",
                        "description": "Send the formatted email report to the farmer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "recipient_email": {
                                    "type": "string",
                                    "description": "Farmer's email address"
                                },
                                "email_content": {
                                    "type": "string",
                                    "description": "Formatted email content"
                                },
                                "subject": {
                                    "type": "string",
                                    "description": "Email subject line"
                                }
                            },
                            "required": ["recipient_email", "email_content", "subject"]
                        }
                    }
                }
            ]
            
            # Create the agent with email expertise
            self.agent = Agent(
                name="AgriPal Email Agent",
                instructions="""
                Expert agricultural report generation and communication specialist.
                
                You are a professional agricultural communications expert responsible for:
                - Creating comprehensive consultation summaries
                - Extracting and prioritizing key agricultural findings
                - Formatting actionable recommendations for farmers
                - Generating professional email reports
                
                Your role is to:
                1. Analyze consultation sessions and extract meaningful insights
                2. Summarize complex agricultural information in farmer-friendly language
                3. Prioritize recommendations by urgency and importance
                4. Generate professional, well-formatted email reports
                5. Ensure all communications are clear, actionable, and supportive
                
                Always provide:
                - Clear, jargon-free summaries accessible to farmers
                - Prioritized, actionable recommendations
                - Professional formatting with proper structure
                - Supportive tone that encourages good farming practices
                - Appropriate disclaimers and follow-up guidance
                
                Focus on creating reports that farmers will find valuable, clear, and actionable.
                """,
                model=self.model,
                tools=tools
            )
            
            logger.info("✅ Email Agent successfully initialized with report generation tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Email Agent: {str(e)}")
            self.agent = None
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Main processing method for email agent using OpenAI Agents SDK
        
        Args:
            message: AgentMessage containing session data for report generation
            
        Returns:
            AgentResponse with email sending results
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"📧 Processing email report request for session {message.session_id}")
            
            if not self.agent:
                raise ValueError("OpenAI Email Agent not initialized")
            
            # Extract session data
            session_data = message.content.get('session_data', {})
            recipient_email = message.content.get('recipient_email')
            user_context = message.content.get('user_context', {})
            
            if not recipient_email:
                raise ValueError("No recipient email provided")
            
            if not session_data:
                raise ValueError("No session data provided for report generation")
            
            # Create agent prompt for report generation
            agent_prompt = self._build_agent_prompt(session_data, user_context, recipient_email)
            
            # Prefer SDK only if run() exists; otherwise, perform direct send
            if hasattr(self.agent, "run"):
                run_result = await self.agent.run(
                    message=agent_prompt,
                    tools_context={
                        "session_data": session_data,
                        "recipient_email": recipient_email,
                        "user_context": user_context,
                        "session_id": message.session_id
                    }
                )
                # Parse agent response into email sending result
                email_result = await self._parse_agent_response(run_result, recipient_email)
            else:
                logger.info("ℹ️ Email agent does not support run(); using direct send flow")
                # Minimal direct composition and send
                report_data = {
                    "user_name": user_context.get("user_name"),
                    "location": user_context.get("location"),
                    "session_id": message.session_id,
                    "session_summary": "Automated summary (direct mode)",
                    "key_findings": [],
                    "recommendations": [],
                    "weather_data": None,
                    "images_analyzed": session_data.get("images_count", 0)
                }
                send_result = await self._tool_send_email_report(
                    recipient_email=recipient_email,
                    report_data=report_data,
                    subject=None,
                    attachments=None
                )
                import json as _json
                parsed = _json.loads(send_result)
                from ..models import EmailSendResult
                email_result = EmailSendResult(
                    success=parsed.get("success", False),
                    message_id=parsed.get("message_id"),
                    recipient=recipient_email,
                    error=parsed.get("error")
                )
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return AgentResponse(
                agent_type=self.agent_type,
                session_id=message.session_id,
                success=True,
                result=email_result.dict(),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"❌ Email report generation failed: {str(e)}")
            
            return AgentResponse(
                agent_type=self.agent_type,
                session_id=message.session_id,
                success=False,
                result={},
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def _build_agent_prompt(
        self, 
        session_data: Dict[str, Any], 
        user_context: Dict[str, Any], 
        recipient_email: str
    ) -> str:
        """
        Build agent prompt for OpenAI Agents SDK
        
        Args:
            session_data: Complete session data
            user_context: User profile and context
            recipient_email: Email recipient
            
        Returns:
            Formatted prompt for the agent
        """
        user_name = user_context.get('user_name', 'Farmer')
        location = user_context.get('location', 'Not specified')
        
        return f"""
        Please generate a comprehensive agricultural consultation report for this farmer.
        
        Farmer Details:
        - Name: {user_name}
        - Email: {recipient_email}
        - Location: {location}
        
        Session Information:
        - Session messages: {len(session_data.get('messages', []))} messages
        - Analysis results available: {bool(session_data.get('analysis_results'))}
        - Images analyzed: {session_data.get('images_count', 0)}
        
        Use the available tools to:
        1. Generate an intelligent summary of the consultation session
        2. Extract and prioritize key findings from the analysis
        3. Format actionable recommendations with appropriate urgency levels
        4. Send a professional email report to the farmer
        
        Ensure the report is:
        - Clear and easy to understand for farmers
        - Well-structured with proper sections
        - Actionable with specific next steps
        - Professional yet supportive in tone
        - Include appropriate disclaimers
        - Use relevant emojis to make it more engaging and visually appealing
        - Include 2-3 emojis per section that relate to the content (🌾🌱🌿💧☀️🌧️🐛🔬📊✅❌⚠️💡🎯🚀)
        """
    
    async def _parse_agent_response(self, run_result: Any, recipient_email: str) -> EmailSendResult:
        """
        Parse OpenAI Agent response into EmailSendResult
        
        Args:
            run_result: Result from agent run
            recipient_email: Email recipient
            
        Returns:
            EmailSendResult with sending status
        """
        try:
            # Check if email was sent successfully through tools
            email_sent = False
            message_id = None
            error_message = None
            
            if hasattr(run_result, 'tool_calls') and run_result.tool_calls:
                for tool_call in run_result.tool_calls:
                    if tool_call.function.name == "send_email_report":
                        tool_result = json.loads(tool_call.function.result) if tool_call.function.result else {}
                        email_sent = tool_result.get('success', False)
                        message_id = tool_result.get('message_id')
                        error_message = tool_result.get('error')
            
            return EmailSendResult(
                success=email_sent,
                message_id=message_id,
                recipient=recipient_email,
                error=error_message
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse email agent response: {str(e)}")
            
            return EmailSendResult(
                success=False,
                message_id=None,
                recipient=recipient_email,
                error=f"Failed to parse agent response: {str(e)}"
            )
    
    # Tool functions for OpenAI Agents SDK
    async def _tool_generate_session_summary(
        self, 
        session_messages: List[Dict], 
        analysis_results: Dict = None, 
        user_context: Dict = None
    ) -> str:
        """
        Tool function for generating intelligent session summaries
        
        Args:
            session_messages: List of session messages
            analysis_results: Optional analysis results
            user_context: Optional user context
            
        Returns:
            JSON string with session summary
        """
        try:
            # Extract key information from messages
            user_messages = [msg for msg in session_messages if msg.get('message_type') == 'user']
            assistant_messages = [msg for msg in session_messages if msg.get('message_type') == 'assistant']
            
            # Build summary prompt
            summary_prompt = f"""
            Summarize this agricultural consultation session in a clear, professional manner.
            
            User queries ({len(user_messages)} messages):
            {' '.join([msg.get('content', '')[:100] + '...' for msg in user_messages[:3]])}
            
            AI responses ({len(assistant_messages)} messages):
            {' '.join([msg.get('content', '')[:100] + '...' for msg in assistant_messages[:3]])}
            
            Analysis results: {bool(analysis_results)}
            User location: {user_context.get('location', 'Not specified') if user_context else 'Not specified'}
            
            Create a concise, farmer-friendly summary highlighting:
            1. The main concerns or questions raised
            2. Key findings and analysis performed
            3. Overall consultation outcome
            4. Areas of focus for recommendations
            """
            
            # Generate summary using AI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content
            
            return json.dumps({
                "summary": summary,
                "total_messages": len(session_messages),
                "user_queries": len(user_messages),
                "ai_responses": len(assistant_messages)
            })
            
        except Exception as e:
            logger.error(f"❌ Session summary generation failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "summary": "Unable to generate session summary",
                "total_messages": len(session_messages) if session_messages else 0
            })
    
    async def _tool_extract_key_findings(
        self, 
        session_data: Dict, 
        perception_results: Dict = None, 
        knowledge_results: Dict = None
    ) -> str:
        """
        Tool function for extracting key findings from consultation
        
        Args:
            session_data: Complete session data
            perception_results: Optional perception analysis results
            knowledge_results: Optional knowledge search results
            
        Returns:
            JSON string with key findings
        """
        try:
            findings = []
            
            # Extract findings from perception results
            if perception_results:
                health_score = perception_results.get('crop_health_score')
                if health_score is not None:
                    findings.append(f"Crop health assessment: {health_score}/100")
                
                issues = perception_results.get('detected_issues', [])
                for issue in issues[:3]:  # Top 3 issues
                    findings.append(f"Visual analysis detected: {issue}")
            
            # Extract findings from knowledge results
            if knowledge_results:
                advice = knowledge_results.get('contextual_advice', '')
                if advice:
                    # Extract key points from advice
                    key_points = advice.split('\n')[:3]  # First 3 points
                    findings.extend([point.strip() for point in key_points if point.strip()])
            
            # Extract from session messages if other sources are limited
            if len(findings) < 3:
                messages = session_data.get('messages', [])
                assistant_messages = [msg for msg in messages if msg.get('message_type') == 'assistant']
                for msg in assistant_messages[:2]:
                    content = msg.get('content', '')
                    if content and len(content) > 50:
                        findings.append(content[:150] + "...")
            
            return json.dumps({
                "key_findings": findings[:5],  # Limit to top 5 findings
                "total_findings": len(findings)
            })
            
        except Exception as e:
            logger.error(f"❌ Key findings extraction failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "key_findings": [],
                "total_findings": 0
            })
    
    async def _tool_format_recommendations(
        self, 
        raw_recommendations: List[str], 
        urgency_levels: List[str] = None, 
        user_context: Dict = None  # Currently unused but may be used for future personalization
    ) -> str:
        """
        Tool function for formatting and prioritizing recommendations
        
        Args:
            raw_recommendations: List of raw recommendations
            urgency_levels: Optional urgency levels
            user_context: Optional user context for personalization
            
        Returns:
            JSON string with formatted recommendations
        """
        try:
            formatted_recommendations = []
            
            if not urgency_levels:
                urgency_levels = ['medium'] * len(raw_recommendations)
            
            # Ensure equal lengths
            min_length = min(len(raw_recommendations), len(urgency_levels))
            
            for i in range(min_length):
                rec = raw_recommendations[i]
                urgency = urgency_levels[i] if i < len(urgency_levels) else 'medium'
                
                # Add urgency prefix
                if urgency == 'high':
                    prefix = "🔴 URGENT: "
                elif urgency == 'medium':
                    prefix = "🟡 Important: "
                else:
                    prefix = "🟢 Consider: "
                
                formatted_rec = f"{prefix}{rec}"
                formatted_recommendations.append({
                    "text": formatted_rec,
                    "urgency": urgency,
                    "original": rec
                })
            
            # Sort by urgency
            urgency_order = {'high': 0, 'medium': 1, 'low': 2}
            formatted_recommendations.sort(key=lambda x: urgency_order.get(x['urgency'], 1))
            
            return json.dumps({
                "formatted_recommendations": [rec['text'] for rec in formatted_recommendations],
                "urgency_breakdown": {
                    "high": sum(1 for rec in formatted_recommendations if rec['urgency'] == 'high'),
                    "medium": sum(1 for rec in formatted_recommendations if rec['urgency'] == 'medium'),
                    "low": sum(1 for rec in formatted_recommendations if rec['urgency'] == 'low')
                },
                "total_recommendations": len(formatted_recommendations)
            })
            
        except Exception as e:
            logger.error(f"❌ Recommendation formatting failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "formatted_recommendations": raw_recommendations,
                "total_recommendations": len(raw_recommendations) if raw_recommendations else 0
            })
    
    async def _tool_send_email_report(
        self, 
        recipient_email: str, 
        report_data: Dict, 
        subject: str = None, 
        attachments: List = None
    ) -> str:
        """
        Tool function for sending email reports
        
        Args:
            recipient_email: Email recipient
            report_data: Complete report data
            subject: Optional custom subject
            attachments: Optional file attachments
            
        Returns:
            JSON string with sending result
        """
        try:
            # Generate default subject if not provided
            if not subject:
                user_name = report_data.get('user_name', 'Farmer')
                date = datetime.utcnow().strftime('%Y-%m-%d')
                subject = f"🌾 AgriPal Consultation Report - {user_name} - {date}"
            
            # Render email content using template
            template = self.templates.get('session_report')
            if not template:
                raise ValueError("Email template not available")
            
            # Prepare template data
            template_data = {
                'user_name': report_data.get('user_name', 'Farmer'),
                'user_email': recipient_email,
                'location': report_data.get('location', 'Not specified'),
                'report_date': datetime.utcnow().strftime('%B %d, %Y'),
                'session_id': report_data.get('session_id', 'Unknown'),
                'session_summary': report_data.get('session_summary', 'No summary available'),
                'key_findings': report_data.get('key_findings', []),
                'recommendations': report_data.get('recommendations', []),
                'weather_data': report_data.get('weather_data'),
                'images_analyzed': report_data.get('images_analyzed', 0)
            }
            
            html_content = template.render(**template_data)
            
            # Send email via SendGrid
            if not self.sendgrid_client:
                raise ValueError("SendGrid client not initialized")
            
            result = await self._send_via_sendgrid(recipient_email, subject, html_content, attachments)
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"❌ Email sending failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "message_id": None
            })
    
    async def _send_via_sendgrid(
        self, 
        recipient_email: str, 
        subject: str, 
        html_content: str, 
        attachments: List = None
    ) -> Dict[str, Any]:
        """
        Send email via SendGrid
        """
        try:
            mail = Mail(
                from_email=(self.from_email, self.from_name),
                to_emails=recipient_email,
                subject=subject,
                html_content=html_content
            )
            
            # Add attachments if provided
            if attachments:
                # TODO: Add attachment logic here
                pass
            
            response = self.sendgrid_client.send(mail)
            
            return {
                "success": True,
                "message_id": response.headers.get('X-Message-Id'),
                "status_code": response.status_code
            }
            
        except Exception as e:
            logger.error(f"❌ SendGrid email failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message_id": None
            }
    
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health status of email agent
        
        Returns:
            Dictionary with health check results
        """
        checks = {
            "openai_api": False,
            "openai_agent": False,
            "sendgrid_client": False,
            "templates_loaded": False
        }
        
        try:
            # Test OpenAI API connection
            await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            checks["openai_api"] = True
            
            # Test OpenAI Agent
            if self.agent is not None:
                checks["openai_agent"] = True
            
            # Test SendGrid client
            if self.sendgrid_client is not None:
                checks["sendgrid_client"] = True
            
            # Test templates
            if self.templates.get('session_report'):
                checks["templates_loaded"] = True
            
        except Exception as e:
            logger.error(f"❌ Email agent health check failed: {str(e)}")
        
        return checks
    
    async def cleanup(self):
        """
        Cleanup resources
        """
        # No specific cleanup needed for email clients
        logger.info("🧹 Email Agent cleanup completed")

