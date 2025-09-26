"""
🌾 AgriPal Gradio UI - Main Interface Implementation
Frontend interface for the AgriPal AI agricultural assistant system.
"""

import gradio as gr
import aiohttp
import json
import logging
import os
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriPalInterface:
    """
    🌾 Main AgriPal Gradio Interface
    
    Provides a user-friendly chat interface for farmers to interact with
    the AI-powered agricultural assistance system.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000/api/v1/agents"):
        self.api_base_url = api_base_url
        self.session_id = None
        self.chat_history = []
        self.user_context = {}
        self.storage_file = "frontend/agripal_session.json"
        
        # Load existing session on startup
        self.load_session()
        
        # UI Components (will be initialized in create_interface)
        self.chatbot = None
        self.message_input = None
        # New compact input bar controls
        self.upload_btn = None
        self.image_gallery = None
        self.image_state = None
        self.location_toggle_btn = None
        self.location = None
        self.location_visible_state = None
        self.send_btn = None
        self.clear_btn = None
        self.export_btn = None
        self.typing_indicator = None
        self.status_indicator = None
    
    def load_session(self):
        """Load session data from storage file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    self.session_id = session_data.get('session_id')
                    self.chat_history = session_data.get('chat_history', [])
                    self.user_context = session_data.get('user_context', {})
                    logger.info(f"📂 Loaded session: {self.session_id} with {len(self.chat_history)} messages")
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.warning(f"⚠️ Could not load session: {e}")
            self.session_id = None
            self.chat_history = []
            self.user_context = {}
    
    def save_session(self):
        """Save session data to storage file with enhanced context"""
        try:
            # Extract additional context from recent conversations
            farming_context = self._extract_farming_context_from_history()
            
            session_data = {
                'session_id': self.session_id,
                'chat_history': self.chat_history,
                'user_context': self.user_context,
                'farming_context': farming_context,
                'last_updated': datetime.now().isoformat(),
                'session_stats': {
                    'total_messages': len(self.chat_history),
                    'session_duration_minutes': self._calculate_session_duration()
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved session: {self.session_id}")
        except (OSError, IOError) as e:
            logger.error(f"❌ Could not save session: {e}")
    
    def _extract_farming_context_from_history(self) -> Dict[str, Any]:
        """Extract farming context from chat history for persistence"""
        context = {
            "crops_mentioned": set(),
            "problems_discussed": set(),
            "solutions_tried": set(),
            "locations_mentioned": set(),
            "frequent_topics": {}
        }
        
        try:
            # Analyze chat history
            for exchange in self.chat_history:
                if len(exchange) >= 2:
                    user_message = exchange[0].lower()
                    assistant_message = exchange[1].lower()
                    
                    # Extract crops
                    crop_keywords = ["corn", "maize", "rice", "cassava", "yam", "beans", "tomato", "pepper", "okra"]
                    for crop in crop_keywords:
                        if crop in user_message or crop in assistant_message:
                            context["crops_mentioned"].add(crop)
                    
                    # Extract problems
                    problem_keywords = ["pest", "disease", "yellow", "wilting", "drought", "fungus", "insect"]
                    for problem in problem_keywords:
                        if problem in user_message:
                            context["problems_discussed"].add(problem)
                    
                    # Extract solutions
                    solution_keywords = ["fertilizer", "spray", "water", "treatment", "pesticide", "fungicide"]
                    for solution in solution_keywords:
                        if solution in assistant_message:
                            context["solutions_tried"].add(solution)
                    
                    # Extract locations
                    location_keywords = ["port harcourt", "lagos", "abuja", "nigeria"]
                    for location in location_keywords:
                        if location in user_message:
                            context["locations_mentioned"].add(location)
            
            # Convert sets to lists for JSON serialization
            context["crops_mentioned"] = list(context["crops_mentioned"])
            context["problems_discussed"] = list(context["problems_discussed"])
            context["solutions_tried"] = list(context["solutions_tried"])
            context["locations_mentioned"] = list(context["locations_mentioned"])
            
        except Exception as e:
            logger.warning(f"⚠️ Could not extract farming context: {e}")
        
        return context
    
    def _calculate_session_duration(self) -> int:
        """Calculate approximate session duration in minutes"""
        try:
            if len(self.chat_history) > 0:
                # Rough estimate based on number of exchanges (assume 2-3 minutes per exchange)
                return len(self.chat_history) * 2
            return 0
        except Exception:
            return 0
    
    def clear_session_storage(self):
        """Clear session data from storage"""
        try:
            if os.path.exists(self.storage_file):
                os.remove(self.storage_file)
                logger.info("🗑️ Cleared session storage")
        except OSError as e:
            logger.error(f"❌ Could not clear session storage: {e}")
    
    def load_conversation_history(self):
        """Load conversation history from backend API"""
        if not self.session_id:
            return
        
        try:
            import asyncio
            
            async def _load_history_async():
                try:
                    url = f"{self.api_base_url.replace('/api/v1/agents', '')}/sessions/{self.session_id}/history"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                history_data = await response.json()
                                messages = history_data.get("messages", [])
                                # Convert backend format to frontend format
                                self.chat_history = []
                                current_user_msg = ""
                                current_ai_msg = ""
                                
                                for msg in messages:
                                    if msg.get("type") == "user":
                                        # If we have a previous pair, save it
                                        if current_user_msg and current_ai_msg:
                                            self.chat_history.append([current_user_msg, current_ai_msg])
                                        current_user_msg = msg.get("content", "")
                                        current_ai_msg = ""
                                    elif msg.get("type") == "assistant":
                                        current_ai_msg = msg.get("content", "")
                                
                                # Add the last pair if it exists
                                if current_user_msg and current_ai_msg:
                                    self.chat_history.append([current_user_msg, current_ai_msg])
                                
                                logger.info(f"📚 Loaded {len(self.chat_history)} message pairs from backend")
                            else:
                                logger.warning(f"⚠️ Could not load history: HTTP {response.status}")
                except Exception as e:
                    logger.error(f"❌ Error loading conversation history: {e}")
            
            # Run the async function
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _load_history_async())
                    future.result()
            except RuntimeError:
                # No event loop running, create a new one
                asyncio.run(_load_history_async())
                
        except Exception as e:
            logger.error(f"❌ Error in load_conversation_history: {e}")
        
    def get_custom_css(self) -> str:
        """Return custom CSS for styling"""
        return """
        /* AgriPal Custom Styles */
        :root {
            --primary-green: #2d5016;
            --secondary-green: #4a7c59;
            --accent-green: #6b8e23;
            --light-green: #f0f8e8;
            --earth-brown: #8b4513;
            --sky-blue: #87ceeb;
            --accent-blue: #2e7bb4;
            --bar-bg: #0f1a14;
            --bar-border: #2a4632;
        }
        
        * { font-family: Georgia, 'Times New Roman', serif; }
        body, input, textarea, button { font-family: Georgia, 'Times New Roman', serif; }
        
        /* Header styling */
        .agripal-header {
            background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .agripal-header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .agripal-header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        
        /* Chat styling */
        .chatbot {
            border: 2px solid var(--light-green);
            border-radius: 10px;
            background: white;
        }
        
        /* Button styling */
        .primary-button {
            background: linear-gradient(135deg, var(--primary-green), var(--secondary-green));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .primary-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3);
        }
        
        /* Compact chat input bar */
        .agripal-input-wrap { margin-top: 16px; }
        .agripal-input-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            background: linear-gradient(135deg, rgba(46,123,180,0.08), rgba(75,124,89,0.08));
            border: 1px solid var(--bar-border);
            border-radius: 999px;
            padding: 8px 12px;
        }
        .agripal-icon-btn button {
            background: transparent !important;
            border: none !important;
            color: var(--secondary-green) !important;
            font-size: 18px !important;
        }
        .agripal-icon-btn button:hover { color: var(--accent-blue) !important; }
        .agripal-message-input textarea, .agripal-message-input input {
            background: transparent !important;
            border: none !important;
            outline: none !important;
        }
        .agripal-send-btn button {
            background: linear-gradient(135deg, var(--primary-green), var(--secondary-green)) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 8px 16px !important;
            font-weight: 700 !important;
        }
        .agripal-inline-row { margin-top: 10px; }
        .agripal-gallery .grid-wrap { gap: 6px !important; }
        .agripal-gallery img { border-radius: 8px; }
        .agripal-loc input { background: #ffffff !important; }
        
        /* Status indicator */
        .status-indicator {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .status-online {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-offline {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .agripal-header h1 {
                font-size: 2em;
            }
            
            .input-section {
                padding: 15px;
            }
        }
        
        """
    
    def create_header(self):
        """Create the header section"""
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="agripal-header">
                    <h1>🌾 AgriPal Assistant</h1>
                    <p>Your AI-powered agricultural companion for crop analysis and farming guidance</p>
                </div>
                """)
                
                # Status indicator
                session_status = ""
                if self.session_id:
                    session_status = f" | Session: {self.session_id[:8]}... | {len(self.chat_history)} messages loaded"
                
                self.status_indicator = gr.HTML(
                    value=f'<div class="status-indicator status-online">🟢 System Online - Ready to help!{session_status}</div>',
                    visible=True
                )
    
    def create_chat_interface(self):
        """Create the main chat interface"""
        with gr.Row():
            with gr.Column(scale=4):
                # Chat history display
                self.chatbot = gr.Chatbot(
                    label="🌾 AgriPal Assistant",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                    avatar_images=(
                        "👨‍🌾",  # User avatar
                        "🌾"     # AgriPal avatar
                    ),
                    placeholder="Welcome to AgriPal! Ask me about your crops, upload field images for analysis, or get farming advice. How can I help you today?",
                    value=self.chat_history  # Load existing chat history
                )
                
                # Typing indicator
                self.typing_indicator = gr.HTML(
                    value="",
                    visible=False
                )
    
    def create_input_controls(self):
        """Create new compact chat-style input controls"""
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="agripal-input-wrap">')

                # Primary compact bar
                with gr.Row(elem_classes=["agripal-input-bar"], equal_height=True):
                    # Location toggle
                    self.location_toggle_btn = gr.Button("@", elem_classes=["agripal-icon-btn"], variant="secondary")

                    # Message input (single line)
                    self.message_input = gr.Textbox(
                        placeholder="Ask about crops, upload images, or request advice...",
                        lines=1,
                        max_lines=4,
                        container=True,
                        scale=4,
                        elem_classes=["agripal-message-input"]
                    )

                    # Upload button (images)
                    self.upload_btn = gr.UploadButton("➕", file_types=["image"], file_count="multiple", elem_classes=["agripal-icon-btn"], variant="secondary")

                    # Send button
                    self.send_btn = gr.Button("Send", elem_classes=["agripal-send-btn"], variant="primary")

                # Inline expanding row
                with gr.Row(elem_classes=["agripal-inline-row"]):
                    # Inline location input (hidden by default)
                    self.location = gr.Textbox(
                        label=None,
                        placeholder="City, State/Province, Country",
                        lines=1,
                        visible=False,
                        elem_classes=["agripal-loc"],
                        scale=2
                    )

                # Thumbnails preview
                with gr.Row():
                    self.image_gallery = gr.Gallery(
                        show_label=False,
                        height=80,
                        columns=8,
                        visible=False,
                        elem_classes=["agripal-gallery"],
                    )

                # Hidden states
                self.image_state = gr.State([])
                self.location_visible_state = gr.State(False)


                # Secondary actions row
                with gr.Row():
                    self.clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    self.export_btn = gr.Button("📧 Export Report", variant="secondary")

                gr.HTML('</div>')
    
    def create_sidebar(self):
        """Create sidebar with additional features"""
        with gr.Column(scale=1, visible=False):  # Hidden by default, can be toggled
            with gr.Group():
                gr.Markdown("### 📚 Quick Actions")
                
                quick_actions = [
                    "🌱 Plant disease diagnosis",
                    "🐛 Pest identification", 
                    "💧 Irrigation advice",
                    "🌾 Harvest timing",
                    "🌿 Soil health check",
                    "📊 Yield optimization"
                ]
                
                for action in quick_actions:
                    gr.Button(action, variant="secondary", size="sm")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                gr.Checkbox(label="🔔 Enable notifications")
                gr.Checkbox(label="🌙 Dark mode")
                gr.Dropdown(label="🌍 Language", choices=["English", "Spanish", "French"])
    
    def create_interface(self):
        """Create the main Gradio interface"""
        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="green",
                secondary_hue="blue", 
                neutral_hue="gray"
            ),
            title="AgriPal - AI Agricultural Assistant",
            css=self.get_custom_css()
        ) as interface:
            
            # Header
            self.create_header()
            
            # Main chat area
            self.create_chat_interface()
            
            # Input controls
            self.create_input_controls()
            
            # Sidebar (optional)
            self.create_sidebar()
            
            # Event handlers
            self.setup_event_handlers()
            
            # Load conversation history from backend if session exists
            if self.session_id:
                self.load_conversation_history()
            
        return interface
    
    def setup_event_handlers(self):
        """Setup event handlers for UI interactions"""
        
        # Upload handler → persist uploads and update gallery/state
        def _on_files_uploaded(files, current_list):
            files = files or []

            # Ensure a persistent uploads directory so files survive after the event
            uploads_dir = os.path.join("frontend", "uploads")
            try:
                os.makedirs(uploads_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Could not create uploads directory: {e}")

            def _persist_file_like(data_bytes: bytes, suggested_name: str | None) -> str | None:
                try:
                    base = os.path.basename(suggested_name or "upload.jpg")
                    name, ext = os.path.splitext(base)
                    if not ext:
                        ext = ".jpg"
                    from datetime import datetime as _dt
                    safe_name = f"{name}_{int(_dt.now().timestamp()*1000)}{ext}"
                    abs_path = os.path.join(uploads_dir, safe_name)
                    with open(abs_path, "wb") as wf:
                        wf.write(data_bytes)
                    return abs_path
                except Exception as e:
                    logger.error(f"❌ Failed persisting uploaded data: {e}")
                    return None

            def _copy_path(src_path: str) -> str | None:
                try:
                    import shutil
                    if not os.path.exists(src_path):
                        return None
                    base = os.path.basename(src_path)
                    name, ext = os.path.splitext(base)
                    if not ext:
                        ext = ".jpg"
                    from datetime import datetime as _dt
                    dst = os.path.join(uploads_dir, f"{name}_{int(_dt.now().timestamp()*1000)}{ext}")
                    shutil.copy2(src_path, dst)
                    return dst
                except Exception as e:
                    logger.error(f"❌ Failed copying uploaded file: {e}")
                    return None

            persisted_paths = []
            for f in files:
                try:
                    # Case 1: Gradio FileData dict
                    if isinstance(f, dict):
                        if f.get("path") and os.path.exists(f["path"]):
                            p = _copy_path(f["path"])  # copy to persistent location
                            if p:
                                persisted_paths.append(p)
                            continue
                        data = f.get("data")
                        if isinstance(data, (bytes, bytearray)):
                            p = _persist_file_like(bytes(data), f.get("name") or None)
                            if p:
                                persisted_paths.append(p)
                            continue
                        # Fallback: name-only; nothing to persist
                        n = f.get("name")
                        if n and os.path.exists(n):
                            p = _copy_path(n)
                            if p:
                                persisted_paths.append(p)
                            continue

                    # Case 2: file-like object
                    if hasattr(f, "read"):
                        try:
                            # Try to read bytes
                            pos = None
                            if hasattr(f, "seek"):
                                try:
                                    pos = f.seek(0)
                                except Exception:
                                    pos = None
                            data_bytes = f.read()
                            if isinstance(data_bytes, (bytes, bytearray)) and len(data_bytes) > 0:
                                p = _persist_file_like(bytes(data_bytes), getattr(f, "name", None))
                                if p:
                                    persisted_paths.append(p)
                                continue
                        except Exception:
                            pass
                        # If it has a filesystem name, try copying
                        n = getattr(f, "name", None)
                        if isinstance(n, str) and os.path.exists(n):
                            p = _copy_path(n)
                            if p:
                                persisted_paths.append(p)
                            continue

                    # Case 3: plain path string
                    if isinstance(f, str):
                        p = _copy_path(f)
                        if p:
                            persisted_paths.append(p)
                        continue

                    logger.warning(f"Skipping unrecognized upload item: {type(f)}")
                except Exception as e:
                    logger.error(f"Error handling uploaded file {f}: {e}")

            existing = [p for p in (current_list or []) if isinstance(p, str) and os.path.exists(p)]
            new_list = existing + persisted_paths
            logger.info(f"📁 Files uploaded: {len(persisted_paths)} new files persisted, total: {len(new_list)}")
            # Show gallery when we have files
            return gr.update(value=new_list, visible=len(new_list) > 0), new_list

        self.upload_btn.upload(
            fn=_on_files_uploaded,
            inputs=[self.upload_btn, self.image_state],
            outputs=[self.image_gallery, self.image_state]
        )

        # Toggle location input visibility
        def _toggle_location(visible: bool):
            new_visible = not bool(visible)
            return new_visible, gr.update(visible=new_visible)

        self.location_toggle_btn.click(
            fn=_toggle_location,
            inputs=[self.location_visible_state],
            outputs=[self.location_visible_state, self.location]
        )


        # Send message handler
        self.send_btn.click(
            fn=self.send_message,
            inputs=[
                self.message_input,
                self.image_state,
                self.location
            ],
            outputs=[
                self.chatbot,
                self.typing_indicator,
                self.send_btn,
                self.message_input,
                self.image_gallery,
                self.image_state
            ],
            show_progress=True
        )
        
        # Enter key handler for message input
        self.message_input.submit(
            fn=self.send_message,
            inputs=[
                self.message_input,
                self.image_state,
                self.location
            ],
            outputs=[
                self.chatbot,
                self.typing_indicator,
                self.send_btn,
                self.message_input,
                self.image_gallery,
                self.image_state
            ],
            show_progress=True
        )

        # Clear chat handler
        self.clear_btn.click(
            fn=self.clear_chat,
            outputs=[self.chatbot, self.message_input, self.image_gallery, self.image_state]
        )
        
        # Export report handler
        self.export_btn.click(
            fn=self.export_report,
            outputs=[gr.File()]
        )
    
    def get_typing_html(self) -> str:
        """Get HTML for typing indicator"""
        return """
        <div style="text-align: center; padding: 10px; color: #666;">
            <span style="animation: pulse 1.5s infinite;">🌾 AgriPal is thinking...</span>
        </div>
        <style>
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """

    def format_image_analysis_markdown(self, image_analysis: dict) -> str:
        """Return a nicely formatted markdown block for image analysis results."""
        try:
            health = image_analysis.get("crop_health_score")
            conf = image_analysis.get("confidence_level")
            issues = image_analysis.get("detected_issues") or []
            severity = image_analysis.get("severity")
            recs = image_analysis.get("recommendations") or []
            metadata = image_analysis.get("metadata") or {}
            observations = metadata.get("observations") or ""
            analysis_text = image_analysis.get("analysis_text") or ""

            parts = []
            parts.append("**Here’s your crop check, nicely summarized:**")

            # Observations
            if observations:
                parts.append(f"- **Visual observations**: {observations}")

            # Health and severity
            if isinstance(health, (int, float)):
                parts.append(f"- **Health score**: {int(round(float(health)))} / 100")
            if severity:
                parts.append(f"- **Severity**: {severity}")

            # Issues
            if issues:
                issues_lines = "\n".join([f"  - {str(i)}" for i in issues])
                parts.append("- **Detected issues**:\n" + issues_lines)

            # Recommendations
            if recs:
                rec_lines = "\n".join([f"  {idx+1}. {str(r)}" for idx, r in enumerate(recs)])
                parts.append("- **Recommendations**:\n" + rec_lines)

            # Confidence
            if isinstance(conf, (int, float)):
                parts.append(f"- **Confidence**: {round(float(conf), 2)}")

            # Friendly analysis paragraph if available
            if analysis_text:
                parts.append("")
                parts.append(analysis_text.strip())

            return "\n".join(parts) if parts else "I've processed your request successfully. The analysis is complete."
        except Exception:
            return "I've processed your request successfully. The analysis is complete."
    
    def send_message(self, message: str, images: List, location: str):
        """Send message to AgriPal backend"""
        import asyncio
        
        async def _send_message_async():
            try:
                # Validate input
                if not message.strip() and not images:
                    return (
                        self.chat_history,
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(value=""),
                        gr.update(visible=False, value=[]),  # Clear image gallery
                        []  # Reset image_state
                    )
                
                # Prepare request data
                request_data = {
                    "query": message.strip(),
                    "location": location,
                    "session_id": self.session_id
                }
                
                # Update user context
                self.user_context.update({
                    "location": location
                })
                
                # Call appropriate API endpoint
                # Check if images exist and are not empty
                has_images = images and len(images) > 0
                logger.info(f"🔍 Image check: images={images}, has_images={has_images}")
                if has_images:
                    # Perception to knowledge workflow with images
                    logger.info(f"📸 Calling perception API with {len(images)} images")
                    response = await self.call_perception_to_knowledge_api(request_data, images)
                else:
                    # Knowledge search only
                    logger.info("📝 Calling knowledge search API (no images)")
                    response = await self.call_knowledge_search_api(request_data)
                
                # Process response
                if response and response.get("status") in ["success", "completed"]:
                    result = response.get("results", {})
                    
                    # Debug logging to understand response structure
                    logger.info("🔍 Response structure: %s", json.dumps(response, indent=2, default=str))
                    
                    # Prefer unified display_text if provided by coordinator OR directly at top-level
                    unified_text = result.get("display_text") or response.get("display_text")
                    if isinstance(unified_text, str) and unified_text.strip():
                        ai_response = unified_text.strip()
                        # If we have display_text, use it as-is and skip all bullet formatting
                    else:
                        # Fallback chain only when no display_text
                        ai_response = None
                        if images:
                            # Prefer the full formatted analysis text from nested image_analysis
                            perception = result.get("perception", {}) or {}
                            image_analysis = perception.get("image_analysis") or {}
                            ai_response = (
                                perception.get("analysis_text") or
                                image_analysis.get("analysis_text") or
                                perception.get("analysis")
                            )
                            # If no pre-formatted text, the backend should handle synthesis
                            # Frontend no longer needs to format raw image analysis data
                        if ai_response is None:
                            ai_response = (
                                result.get("knowledge", {}).get("contextual_advice") or
                                result.get("response") or
                                result.get("analysis", {}).get("summary") or
                                str(result.get("knowledge") or result.get("perception") or result)
                            )
                    
                    # Convert lists to natural text without forced formatting
                    if isinstance(ai_response, list):
                        # Simply join list items naturally without forcing bullet points
                        ai_response = "\n".join([str(item) for item in ai_response])
                    
                    if not ai_response or ai_response == "I've analyzed your request. Here's what I found: {}":
                        ai_response = "I've processed your request successfully. The analysis is complete."
                    
                    # Add to chat history with images if present
                    if has_images and images:
                        # Include image information in the user message for display
                        image_info = f"📸 Uploaded {len(images)} image(s)"
                        if len(images) == 1:
                            image_info = "📸 Uploaded image"
                        enhanced_message = f"{message}\n\n{image_info}"
                    else:
                        enhanced_message = message
                    
                    self.chat_history.append([enhanced_message, ai_response])
                    
                    # Update session ID
                    self.session_id = response.get("session_id", self.session_id)
                    
                    # Save session after successful message
                    self.save_session()
                    
                else:
                    # Debug: Log the actual response to understand what's happening
                    logger.error("❌ Response not successful. Status: %s", response.get('status') if response else 'None')
                    logger.error("❌ Full response: %s", json.dumps(response, indent=2, default=str) if response else 'No response')
                    
                    error_msg = response.get("error", "An error occurred") if response else "Connection failed"
                    # Include image info in error message if images were uploaded
                    if has_images and images:
                        image_info = f"📸 Uploaded {len(images)} image(s)"
                        if len(images) == 1:
                            image_info = "📸 Uploaded image"
                        enhanced_message = f"{message}\n\n{image_info}"
                    else:
                        enhanced_message = message
                    self.chat_history.append([enhanced_message, f"❌ Error: {error_msg}"])
                
                # Return final result - clear images after sending (like ChatGPT)
                return (
                    self.chat_history,
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(value=""),
                    gr.update(visible=False, value=[]),  # Clear image gallery
                    []  # Reset image_state
                )
                
            except (aiohttp.ClientError, json.JSONDecodeError, ValueError) as e:
                logger.error("Error in send_message: %s", str(e))
                error_msg = f"❌ Connection error: {str(e)}"
                # Include image info in error message if images were uploaded
                if has_images and images:
                    image_info = f"📸 Uploaded {len(images)} image(s)"
                    if len(images) == 1:
                        image_info = "📸 Uploaded image"
                    enhanced_message = f"{message}\n\n{image_info}"
                else:
                    enhanced_message = message
                self.chat_history.append([enhanced_message, error_msg])
                return (
                    self.chat_history,
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(value=""),
                    gr.update(visible=False, value=[]),  # Clear image gallery on error too
                    []
                )
        
        # Run the async function
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _send_message_async())
                return future.result()
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(_send_message_async())

    
    async def call_perception_to_knowledge_api(self, request_data: dict, images: List) -> dict:
        """Call the perception to knowledge workflow API endpoint"""
        try:
            url = f"{self.api_base_url}/workflows/perception-to-knowledge"
            
            # Prepare form data
            form_data = aiohttp.FormData()
            form_data.add_field('query', request_data.get('query', ''))
            form_data.add_field('location', request_data.get('location', '') or '')
            if request_data.get('session_id'):
                form_data.add_field('session_id', str(request_data.get('session_id')))
            
            # Add images
            for i, image in enumerate(images):
                if not image:  # Skip None or empty images
                    continue
                    
                try:
                    if hasattr(image, 'name') and hasattr(image, 'read'):
                        # File object
                        image.seek(0)  # Reset file pointer
                        image_data = image.read()
                        if image_data:  # Only add if we have data
                            form_data.add_field('files', image_data, filename=f'image_{i}.jpg', content_type='image/jpeg')
                    elif isinstance(image, str) and os.path.exists(image):
                        # File path
                        with open(image, 'rb') as f:
                            image_data = f.read()
                            if image_data:  # Only add if we have data
                                form_data.add_field('files', image_data, filename=f'image_{i}.jpg', content_type='image/jpeg')
                    else:
                        logger.warning(f"Skipping invalid image at index {i}: {image}")
                        continue
                except (FileNotFoundError, PermissionError, OSError) as e:
                    logger.error("Error reading image file %s: %s", image, str(e))
                    continue
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form_data) as response:
                    logger.info("🌐 API Response Status: %s", response.status)
                    if response.status == 200:
                        response_data = await response.json()
                        logger.info("✅ API Response Data: %s", json.dumps(response_data, indent=2, default=str))
                        return response_data
                    else:
                        error_text = await response.text()
                        logger.error("API Error %d: %s", response.status, error_text)
                        return {"status": "error", "error": f"HTTP {response.status}: {error_text}"}
                        
        except (aiohttp.ClientError, FileNotFoundError, OSError) as e:
            logger.error("Error calling perception to knowledge API: %s", str(e))
            return {"status": "error", "error": str(e)}

    async def call_knowledge_search_api(self, request_data: dict) -> dict:
        """Call the knowledge search API endpoint"""
        try:
            url = f"{self.api_base_url}/knowledge/search"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "error", "error": f"HTTP {response.status}"}
                        
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            logger.error("Error calling knowledge search API: %s", str(e))
            return {"status": "error", "error": str(e)}
    
    def clear_chat(self):
        """Clear the chat history"""
        self.chat_history = []
        self.session_id = None
        self.user_context = {}
        self.clear_session_storage()
        return (
            [],
            gr.update(value=""),
            gr.update(visible=False, value=[]),
            []  # Reset image_state
        )
    
    def export_report(self):
        """Export chat history as a report"""
        try:
            if not self.chat_history:
                return None
            
            # Create a simple text report
            report_content = f"""
AgriPal Consultation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.session_id or 'N/A'}

Conversation History:
"""
            
            for i, (user_msg, ai_msg) in enumerate(self.chat_history, 1):
                report_content += f"\n{i}. User: {user_msg}\n   AgriPal: {ai_msg}\n"
            
            # Save to temporary file
            report_path = f"agripal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return report_path
            
        except (OSError, IOError) as e:
            logger.error("Error exporting report: %s", str(e))
            return None
    
    async def check_system_health(self):
        """Check system health and update status indicator"""
        try:
            url = f"{self.api_base_url}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        if health_data.get("healthy", False):
                            return '<div class="status-indicator status-online">🟢 System Online - Ready to help!</div>'
                        else:
                            return '<div class="status-indicator status-offline">🟡 System Limited - Some features unavailable</div>'
                    else:
                        return '<div class="status-indicator status-offline">🔴 System Offline - Please try again later</div>'
        except (aiohttp.ClientError, aiohttp.ServerTimeoutError) as e:
            logger.error("Health check failed: %s", str(e))
            return '<div class="status-indicator status-offline">🔴 System Offline - Please try again later</div>'


def create_agripal_interface(api_base_url: str = "http://localhost:8000/api/v1/agents") -> gr.Blocks:
    """
    Create and return the AgriPal Gradio interface
    
    Args:
        api_base_url: Base URL for the AgriPal API
        
    Returns:
        Gradio Blocks interface
    """
    interface = AgriPalInterface(api_base_url)
    return interface.create_interface()


def main():
    """Main entry point for AgriPal Gradio UI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AgriPal Gradio UI")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1/agents", 
                       help="AgriPal API base URL")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create interface
    app = create_agripal_interface(args.api_url)
    
    # Launch with appropriate settings
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    )


if __name__ == "__main__":
    main()
