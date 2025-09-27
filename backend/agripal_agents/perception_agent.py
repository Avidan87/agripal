"""
🔍 Perception Agent - Visual Analysis and Interpretation
Specialized AI agent for crop image analysis, disease detection, and visual assessment.
Uses OpenAI Agents SDK for enhanced agent capabilities and tool calling.
"""
import base64
import io
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Heavy imports - will be imported when needed
cv2 = None
np = None
Image = None

# OpenAI Agents SDK imports
from openai import AsyncOpenAI
from agents import Agent, Tool

from ..models import (
    AgentMessage, 
    AgentResponse, 
    AgentType, 
    ImageAnalysisResult,
    SeverityLevel
)
from ..config import settings

logger = logging.getLogger(__name__)

def _import_heavy_dependencies():
    """Import heavy dependencies only when needed"""
    global cv2, np, Image
    try:
        import cv2  # type: ignore
        import numpy as np
        from PIL import Image
        return True
    except ImportError as e:
        logger.warning(f"⚠️ Heavy dependencies not available: {e}")
        return False

class PerceptionAgent:
    """
    🔍 Vision-based crop analysis agent using OpenAI Agents SDK
    
    Capabilities:
    - Disease detection and classification
    - Pest identification
    - Crop stress assessment (drought, nutrient deficiency)
    - Growth stage analysis
    - Image quality validation and enhancement
    """
    
    def __init__(self):
        self.agent_type = AgentType.PERCEPTION
        self.model = settings.OPENAI_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Check if heavy dependencies are available
        self.vision_enabled = _import_heavy_dependencies()
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Analysis parameters
        self.max_image_size = 1024  # Max width/height for processing
        self.confidence_threshold = 0.7
        
        # Initialize OpenAI Agent SDK
        self.agent = None
        self._setup_agent()
        
        logger.info("🔍 Perception Agent initialized with OpenAI Agents SDK")
    
    def _setup_agent(self):
        """
        Initialize the OpenAI Agent with tools for crop analysis
        """
        try:
            # Define tools for the perception agent
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_crop_image",
                        "description": "Analyze field images for crop health, diseases, pests, and recommendations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_base64": {
                                    "type": "string",
                                    "description": "Base64 encoded field image"
                                },
                                "crop_type": {
                                    "type": "string",
                                    "description": "Type of crop being analyzed"
                                },
                                "query_context": {
                                    "type": "string",
                                    "description": "User's query or concern about the crop"
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional image metadata"
                                }
                            },
                            "required": ["image_base64"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "validate_image_quality",
                        "description": "Check if uploaded image is suitable for crop analysis",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Raw image data to validate"
                                }
                            },
                            "required": ["image_data"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "enhance_image_quality",
                        "description": "Enhance image quality for better analysis",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Image data to enhance"
                                }
                            },
                            "required": ["image_data"]
                        }
                    }
                }
            ]
            
            # Create the agent with agricultural expertise
            self.agent = Agent(
                name="AgriPal Perception Agent",
                instructions="""
                Expert agricultural visual analysis agent specializing in crop health assessment.
                
                You are an expert agricultural consultant with advanced computer vision capabilities.
                Your role is to analyze field images and provide comprehensive crop health assessments.
                
                Key expertise areas:
                - Disease identification and classification
                - Pest detection and management recommendations
                - Nutrient deficiency diagnosis
                - Growth stage assessment
                - Environmental stress evaluation
                - Actionable farming recommendations
                
                Always provide:
                1. Health score (0-100)
                2. Confidence level (0.0-1.0)
                3. Specific issues identified
                4. Severity assessment
                5. Actionable recommendations
                6. Urgency level for intervention
                
                Be practical, specific, and focus on actionable insights farmers can implement.
                """,
                model=self.model,
                tools=tools
            )
            
            logger.info("✅ OpenAI Agent successfully initialized with agricultural tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI Agent: {str(e)}")
            self.agent = None
    
    async def _fallback_analysis(self, image_base64: str, query_context: str, crop_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback analysis method when agent tools fail
        
        Args:
            image_base64: Base64 encoded image
            query_context: User's query
            crop_type: Type of crop
            metadata: Image metadata
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Use OpenAI Vision API directly for fallback analysis
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyze this agricultural image. Query: {query_context}. Crop type: {crop_type}. Provide analysis in JSON format with detected_issues, crop_health_score (0-100), confidence_level (0-1), and recommendations fields."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to structured response
                return {
                    "detected_issues": ["Analysis completed via fallback method"],
                    "crop_health_score": 75.0,
                    "confidence_level": 0.7,
                    "recommendations": [content[:200] + "..." if len(content) > 200 else content]
                }
                
        except Exception as e:
            logger.error(f"❌ Fallback analysis failed: {str(e)}")
            return {
                "detected_issues": ["Analysis failed"],
                "crop_health_score": 50.0,
                "confidence_level": 0.3,
                "recommendations": ["Unable to analyze image. Please try again or contact support."]
            }
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Main processing method for perception agent using OpenAI Agents SDK
        
        Args:
            message: AgentMessage containing image data and context
            
        Returns:
            AgentResponse with visual analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"🔍 Processing perception request for session {message.session_id}")
            
            if not self.agent:
                raise ValueError("OpenAI Agent not initialized")
            
            # Extract image data and context
            image_data = message.content.get('image')
            query_context = message.content.get('query', '')
            crop_type = message.content.get('crop_type')
            
            if not image_data:
                raise ValueError("No image data provided for perception analysis")
            
            # Preprocess image
            processed_image, metadata = await self._preprocess_image(image_data)
            
            # Convert image to base64 for agent tools
            image_base64 = await self._image_to_base64(processed_image)
            
            # Create agent prompt with context
            agent_prompt = self._build_agent_prompt(query_context, crop_type, metadata)
            
            # Use direct tool call instead of agent.run (due to SDK version compatibility)
            if self.agent:
                # Try to use the agent if available, but fall back to direct tool call
                try:
                    # For now, use direct tool call since agent.run is not available
                    analysis_result = await self._tool_analyze_crop_image(
                        image_base64=image_base64,
                        crop_type=crop_type,
                        query_context=query_context,
                        metadata=metadata
                    )
                    # Parse the JSON result
                    import json
                    analysis_data = json.loads(analysis_result)
                except Exception as e:
                    logger.warning(f"⚠️ Agent tool call failed: {str(e)}, using fallback")
                    analysis_data = await self._fallback_analysis(image_base64, query_context, crop_type, metadata)
            else:
                # Use direct tool call
                analysis_result = await self._tool_analyze_crop_image(
                    image_base64=image_base64,
                    crop_type=crop_type,
                    query_context=query_context,
                    metadata=metadata
                )
                # Parse the JSON result
                import json
                analysis_data = json.loads(analysis_result)
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return AgentResponse(
                agent_type=self.agent_type,
                session_id=message.session_id,
                success=True,
                result={
                    "analysis": analysis_data.get("recommendations", ["Analysis completed successfully"]),
                    "image_analysis": analysis_data,
                    "metadata": metadata
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"❌ Perception analysis failed: {str(e)}")
            
            return AgentResponse(
                agent_type=self.agent_type,
                session_id=message.session_id,
                success=False,
                result={},
                error=str(e),
                processing_time_ms=processing_time
            )
    
    async def _preprocess_image(self, image_data: bytes) -> tuple[Any, Dict[str, Any]]:
        """
        Preprocess and enhance uploaded field image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        if not self.vision_enabled:
            raise ValueError("Vision processing not available - heavy dependencies not installed")
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image format or corrupted image data")
            
            # Extract basic metadata
            height, width = image.shape[:2]
            metadata = {
                "original_size": {"width": width, "height": height},
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "file_size_bytes": len(image_data)
            }
            
            # Validate image quality
            if not self._is_valid_field_image(image):
                logger.warning("⚠️ Image quality validation failed")
                metadata["quality_warning"] = "Low quality image detected"
            
            # Enhance image for better analysis
            enhanced = await self._enhance_image_quality(image)
            metadata["enhanced"] = True
            
            logger.info(f"📸 Image preprocessed: {width}x{height} -> enhanced")
            return enhanced, metadata
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def _is_valid_field_image(self, image: Any) -> bool:
        """
        Validate if image appears to be a valid field/crop image
        
        Args:
            image: OpenCV image array
            
        Returns:
            Boolean indicating if image is valid for analysis
        """
        if not self.vision_enabled:
            return True  # Skip validation if vision not available
        
        # Check image dimensions
        height, width = image.shape[:2]
        if width < 100 or height < 100:
            return False
        
        # Check if image is not too dark or too bright
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 20 or mean_brightness > 240:
            return False
        
        # Check for sufficient detail (not blurry)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Threshold for blur detection
            return False
        
        return True
    
    async def _enhance_image_quality(self, image: Any) -> Any:
        """
        Enhance image quality for better AI analysis
        
        Args:
            image: OpenCV image array
            
        Returns:
            Enhanced image array
        """
        # Noise reduction
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Contrast enhancement using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Resize for optimal processing
        height, width = enhanced.shape[:2]
        if width > self.max_image_size or height > self.max_image_size:
            scale = self.max_image_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return enhanced
    
    def _build_agent_prompt(self, query_context: str, crop_type: Optional[str], metadata: Dict[str, Any]) -> str:
        """
        Build agent prompt for OpenAI Agents SDK
        
        Args:
            query_context: User's query context
            crop_type: Type of crop if known
            metadata: Image metadata
            
        Returns:
            Formatted prompt for the agent
        """
        crop_context = f"This is a {crop_type} crop. " if crop_type else ""
        
        return f"""
        Please analyze the uploaded field image for crop health assessment.
        
        {crop_context}User Query: {query_context}
        
        Image Details: {metadata.get('original_size', 'Unknown size')}
        
        Use the analyze_crop_image tool to perform a comprehensive analysis including:
        1. Visual observations and crop health assessment
        2. Disease, pest, or disorder detection
        3. Nutrient deficiency signs
        4. Environmental stress indicators
        5. Growth stage evaluation
        6. Specific actionable recommendations
        
        Provide practical, farmer-focused insights with urgency indicators.
        """
    
    async def _parse_agent_response(self, run_result: Any, metadata: Dict[str, Any]) -> ImageAnalysisResult:
        """
        Parse OpenAI Agent response into structured ImageAnalysisResult
        
        Args:
            run_result: Result from agent run
            metadata: Image metadata
            
        Returns:
            Structured ImageAnalysisResult
        """
        try:
            # Extract response from agent run
            response_text = run_result.response if hasattr(run_result, 'response') else str(run_result)
            
            # Check if agent used tools and extract tool results
            tool_results = {}
            if hasattr(run_result, 'tool_calls') and run_result.tool_calls:
                for tool_call in run_result.tool_calls:
                    if tool_call.function.name == "analyze_crop_image":
                        tool_results = json.loads(tool_call.function.result) if tool_call.function.result else {}
            
            # Parse structured response (reuse existing parsing logic)
            if tool_results:
                return ImageAnalysisResult(
                    detected_issues=tool_results.get('detected_issues', []),
                    crop_health_score=tool_results.get('crop_health_score', 75.0),
                    confidence_level=tool_results.get('confidence_level', 0.8),
                    recommendations=tool_results.get('recommendations', []),
                    severity=SeverityLevel(tool_results.get('severity', 'low')),
                    metadata={
                        **metadata,
                        "analysis_method": "openai-agents-sdk",
                        "response_text": response_text
                    }
                )
            else:
                # Fallback to text parsing if no tool results
                return await self._parse_analysis_response(response_text, metadata)
                
        except Exception as e:
            logger.error(f"❌ Failed to parse agent response: {str(e)}")
            
            # Generate dynamic expert recommendation for parse error
            from backend.utils.expert_recommendations import expert_system
            expert_rec = expert_system.get_expert_recommendation(
                issue_type="image analysis",
                crop_type=metadata.get('crop_type') if metadata else None,
                location=metadata.get('location') if metadata else None,
                confidence=0.3,
                severity="medium"
            )
            
            recommendations = ["Unable to complete detailed analysis due to technical issues"]
            if expert_rec:
                recommendations.append(expert_rec)
            else:
                recommendations.append("Please try uploading the image again or contact support")
            
            return ImageAnalysisResult(
                detected_issues=["Unable to complete detailed analysis"],
                crop_health_score=50.0,
                confidence_level=0.3,
                recommendations=recommendations,
                severity=SeverityLevel.MEDIUM,
                metadata={**metadata, "parse_error": str(e)}
            )
    
    # Tool functions for OpenAI Agents SDK
    async def _tool_analyze_crop_image(self, image_base64: str, crop_type: str = None, query_context: str = "", metadata: dict = None) -> str:
        """
        Tool function for analyzing crop images using GPT-4o Vision
        
        Args:
            image_base64: Base64 encoded image
            crop_type: Type of crop being analyzed
            query_context: User's query context
            metadata: Additional metadata
            
        Returns:
            JSON string with analysis results
        """
        try:
            # Build analysis prompt for vision model
            prompt = self._build_analysis_prompt(query_context, crop_type)
            
            # Call GPT-4o Vision API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyze this field image. User query: {query_context}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse and structure the response
            structured_result = await self._parse_analysis_response(analysis_text, metadata or {})
            
            # Return as JSON for agent tool
            return json.dumps({
                "detected_issues": structured_result.detected_issues,
                "crop_health_score": structured_result.crop_health_score,
                "confidence_level": structured_result.confidence_level,
                "recommendations": structured_result.recommendations,
                "severity": structured_result.severity.value,
                "analysis_text": analysis_text
            })
            
        except Exception as e:
            logger.error(f"❌ Tool analyze_crop_image failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "detected_issues": ["Analysis failed due to technical error"],
                "crop_health_score": 50.0,
                "confidence_level": 0.0,
                "recommendations": ["Please try uploading the image again"],
                "severity": "medium"
            })
    
    async def _tool_validate_image_quality(self, image_data: str) -> str:
        """
        Tool function for validating image quality
        
        Args:
            image_data: Image data to validate
            
        Returns:
            JSON string with validation results
        """
        try:
            # Convert from base64 if needed
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            is_valid = self._is_valid_field_image(image)
            
            return json.dumps({
                "is_valid": is_valid,
                "dimensions": {"width": image.shape[1], "height": image.shape[0]},
                "message": "Image is suitable for analysis" if is_valid else "Image quality may affect analysis accuracy"
            })
            
        except Exception as e:
            return json.dumps({
                "is_valid": False,
                "error": str(e),
                "message": "Unable to validate image"
            })
    
    async def _tool_enhance_image_quality(self, image_data: str) -> str:
        """
        Tool function for enhancing image quality
        
        Args:
            image_data: Image data to enhance
            
        Returns:
            JSON string with enhancement results
        """
        try:
            # Convert from base64 if needed
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            enhanced = await self._enhance_image_quality(image)
            enhanced_base64 = await self._image_to_base64(enhanced)
            
            return json.dumps({
                "enhanced": True,
                "enhanced_image_base64": enhanced_base64,
                "message": "Image successfully enhanced for better analysis"
            })
            
        except Exception as e:
            return json.dumps({
                "enhanced": False,
                "error": str(e),
                "message": "Unable to enhance image"
            })
    
    async def _analyze_image(
        self, 
        image: Any, 
        query_context: str,
        crop_type: Optional[str],
        metadata: Dict[str, Any]
    ) -> ImageAnalysisResult:
        """
        Perform comprehensive image analysis using GPT-4o Vision
        
        Args:
            image: Preprocessed image array
            query_context: User's query for context
            crop_type: Type of crop being analyzed
            metadata: Image metadata
            
        Returns:
            ImageAnalysisResult with detailed findings
        """
        try:
            # Convert image to base64 for API
            image_base64 = await self._image_to_base64(image)
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(query_context, crop_type)
            
            # Call GPT-4o Vision API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyze this field image. User query: {query_context}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE
            )
            
            # Parse response and extract structured data
            analysis_text = response.choices[0].message.content
            structured_result = await self._parse_analysis_response(analysis_text, metadata)
            
            logger.info(f"✅ Image analysis completed with {structured_result.confidence_level:.2f} confidence")
            return structured_result
            
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {str(e)}")
            # Return a basic result structure even on failure
            return ImageAnalysisResult(
                detected_issues=["Analysis failed due to technical error"],
                crop_health_score=50.0,
                confidence_level=0.0,
                recommendations=["Please try uploading the image again"],
                severity=SeverityLevel.MEDIUM,
                metadata={"error": str(e)}
            )
    
    def _build_analysis_prompt(self, query_context: str, crop_type: Optional[str]) -> str:
        """
        Build comprehensive analysis prompt for GPT-4o Vision
        
        Args:
            query_context: User's query context
            crop_type: Type of crop if known
            
        Returns:
            Formatted prompt string
        """
        crop_context = f"This is a {crop_type} crop. " if crop_type else ""
        
        return f"""
        You are an expert agricultural consultant analyzing field images for crop health assessment.
        
        {crop_context}Please provide a comprehensive analysis that will be used to create natural, 
        engaging responses for farmers. Focus on practical insights and actionable guidance.
        
        ANALYSIS AREAS:
        1. Visual Observations: What do you see in this image?
        2. Health Assessment: Overall crop health score (0-100)
        3. Disease Detection: Any visible diseases, pests, or disorders
        4. Nutrient Status: Signs of nutrient deficiencies or toxicities
        5. Environmental Stress: Drought, heat, cold, or other stress indicators
        6. Growth Stage: Estimated growth stage and development
        7. Recommendations: Specific actionable advice
        8. Urgency Level: How urgent is intervention (low/medium/high/critical)
        
        FORMAT FOR STRUCTURED DATA (for internal processing):
        HEALTH_SCORE: [0-100]
        CONFIDENCE: [0.0-1.0]
        ISSUES: [comma-separated list]
        SEVERITY: [low/medium/high/critical]
        RECOMMENDATIONS: [numbered list]
        OBSERVATIONS: [detailed description]
        
        ANALYSIS_TEXT: [Write a natural, conversational analysis that a farmer would find engaging and helpful. 
        This should be written as if you're speaking directly to the farmer, providing strategic insights 
        and practical guidance. Be encouraging and supportive, especially if there are challenges.
        Use relevant emojis throughout (3-5 emojis) to make it more engaging: 🌾🌱🌿💧☀️🌧️🐛🔬📊✅❌⚠️💡🎯🚀🌿🌽🍅🥕🌶️🥬🌾🌻🌺🌸🌼🌷]
        
        User Context: {query_context}
        
        Focus on providing strategic insights that help farmers make informed decisions about their crops.
        """
    
    async def _image_to_base64(self, image: Any) -> str:
        """
        Convert OpenCV image to base64 string
        
        Args:
            image: OpenCV image array
            
        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
    
    def _generate_analysis_text(self, health_score: float, issues: List[str], severity: SeverityLevel, recommendations: List[str], observations: str) -> str:
        """
        Generate natural analysis text when not provided by GPT-4o
        """
        try:
            # Create a natural, conversational analysis
            analysis_parts = []
            
            # Health assessment
            if health_score >= 80:
                health_desc = "excellent health" 
                emoji = "🌱✅"
            elif health_score >= 60:
                health_desc = "moderate health"
                emoji = "🌿⚠️"
            elif health_score >= 40:
                health_desc = "concerning health"
                emoji = "🌾❌"
            else:
                health_desc = "poor health"
                emoji = "🌾🚨"
            
            analysis_parts.append(f"Your crop shows {health_desc} with a score of {int(health_score)}/100 {emoji}")
            
            # Issues
            if issues:
                if len(issues) == 1:
                    analysis_parts.append(f"I can see signs of {issues[0]}.")
                else:
                    analysis_parts.append(f"I've identified several concerns: {', '.join(issues[:3])}.")
            
            # Severity
            severity_desc = severity.value.lower()
            if severity_desc == "high" or severity_desc == "critical":
                analysis_parts.append(f"The situation requires {severity_desc} attention. 🚨")
            elif severity_desc == "medium":
                analysis_parts.append(f"This is a {severity_desc} concern that should be addressed. ⚠️")
            else:
                analysis_parts.append(f"The issues appear to be {severity_desc} in severity. 💡")
            
            # Observations
            if observations:
                analysis_parts.append(f"Key observations: {observations}")
            
            # Recommendations
            if recommendations:
                analysis_parts.append("Here's what I recommend:")
                for i, rec in enumerate(recommendations[:3], 1):
                    analysis_parts.append(f"{i}. {rec}")
            
            # Encouraging closing
            analysis_parts.append("Remember, early detection and proper care can make a big difference in your crop's success! 🌾💪")
            
            return "\n\n".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate analysis text: {str(e)}")
            return f"Based on the visual analysis, your crop has a health score of {int(health_score)}/100 with {severity.value} severity. I've identified {', '.join(issues[:2]) if issues else 'some concerns'} that need attention."
    
    async def _parse_analysis_response(self, response_text: str, metadata: Dict[str, Any]) -> ImageAnalysisResult:
        """
        Parse GPT-4o response into structured ImageAnalysisResult
        
        Args:
            response_text: Raw response from GPT-4o
            metadata: Image metadata
            
        Returns:
            Structured ImageAnalysisResult
        """
        try:
            lines = response_text.split('\n')
            
            # Initialize default values
            health_score = 75.0
            confidence = 0.8
            issues = []
            severity = SeverityLevel.LOW
            recommendations = []
            observations = ""
            analysis_text = None
            
            # Parse structured response
            for line in lines:
                line = line.strip()
                if line.startswith('HEALTH_SCORE:'):
                    try:
                        health_score = float(line.split(':', 1)[1].strip())
                        health_score = max(0.0, min(100.0, health_score))
                    except (ValueError, IndexError):
                        pass
                        
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except (ValueError, IndexError):
                        pass
                        
                elif line.startswith('ISSUES:'):
                    issues_str = line.split(':', 1)[1].strip()
                    issues = [issue.strip() for issue in issues_str.split(',') if issue.strip()]
                    
                elif line.startswith('SEVERITY:'):
                    severity_str = line.split(':', 1)[1].strip().lower()
                    try:
                        severity = SeverityLevel(severity_str)
                    except ValueError:
                        severity = SeverityLevel.MEDIUM
                        
                elif line.startswith('RECOMMENDATIONS:'):
                    # Start collecting recommendations
                    recommendations = []
                    continue
                    
                elif line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                    # This is a recommendation item
                    rec_text = line.lstrip('123456789.-• ').strip()
                    if rec_text:
                        recommendations.append(rec_text)
                        
                elif line.startswith('OBSERVATIONS:'):
                    observations = line.split(':', 1)[1].strip()
                    
                elif line.startswith('ANALYSIS_TEXT:'):
                    analysis_text = line.split(':', 1)[1].strip()
            
            # If no specific issues detected, infer from health score
            if not issues:
                if health_score < 30:
                    issues = ["Severe crop stress detected"]
                elif health_score < 60:
                    issues = ["Moderate crop health concerns"]
                elif health_score < 80:
                    issues = ["Minor crop health issues"]
                else:
                    issues = ["Crop appears healthy"]
            
            # Ensure we have recommendations
            if not recommendations:
                recommendations = [
                    "Continue monitoring crop development",
                    "Maintain current agricultural practices",
                    "Consider consulting local agricultural extension services"
                ]
            
            # Generate analysis_text if not found in response
            if not analysis_text:
                analysis_text = self._generate_analysis_text(health_score, issues, severity, recommendations, observations)
            
            return ImageAnalysisResult(
                detected_issues=issues,
                crop_health_score=health_score,
                confidence_level=confidence,
                recommendations=recommendations,
                severity=severity,
                analysis_text=analysis_text,
                metadata={
                    **metadata,
                    "analysis_method": "gpt-4o-vision",
                    "response_length": len(response_text),
                    "observations": observations
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse analysis response: {str(e)}")
            
            # Generate dynamic expert recommendation for analysis error
            from backend.utils.expert_recommendations import expert_system
            expert_rec = expert_system.get_expert_recommendation(
                issue_type="image analysis",
                crop_type=metadata.get('crop_type') if metadata else None,
                location=metadata.get('location') if metadata else None,
                confidence=0.3,
                severity="medium"
            )
            
            recommendations = ["Unable to complete detailed analysis due to parsing issues"]
            if expert_rec:
                recommendations.append(expert_rec)
            else:
                recommendations.append("Please try uploading a clearer image or contact support")
            
            # Return safe default result
            return ImageAnalysisResult(
                detected_issues=["Unable to complete detailed analysis"],
                crop_health_score=50.0,
                confidence_level=0.3,
                recommendations=recommendations,
                severity=SeverityLevel.MEDIUM,
                metadata={**metadata, "parse_error": str(e)}
            )
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health status of perception agent with OpenAI Agents SDK
        
        Returns:
            Dictionary with health check results
        """
        checks = {
            "openai_api": False,
            "openai_agent": False,
            "vision_model": False,
            "image_processing": False,
            "agent_tools": False
        }
        
        try:
            # Test OpenAI API connection
            await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for health check
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            checks["openai_api"] = True
            
            # Test OpenAI Agent initialization
            if self.agent is not None:
                checks["openai_agent"] = True
            
            # Test vision model availability
            if self.model in ["gpt-4o", "gpt-4-vision-preview"]:
                checks["vision_model"] = True
            
            # Test image processing capabilities
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            await self._enhance_image_quality(test_image)
            checks["image_processing"] = True
            
            # Test agent tools
            if self.agent and hasattr(self.agent, 'tools') and self.agent.tools:
                checks["agent_tools"] = len(self.agent.tools) > 0
            
        except Exception as e:
            logger.error(f"❌ Perception agent health check failed: {str(e)}")
        
        return checks
    
    async def cleanup(self):
        """Cleanup perception agent resources"""
        try:
            # Close any open connections or resources
            if hasattr(self, 'client') and self.client:
                # OpenAI client doesn't need explicit cleanup
                pass
            
            logger.info("🧹 Perception Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Perception Agent cleanup failed: {str(e)}")