"""
🧠 AgriPal In-Memory Conversation Storage
Fallback storage for conversation history when database is unavailable.
"""
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class InMemoryConversationStorage:
    """In-memory storage for conversation history as fallback"""
    
    def __init__(self, storage_file: str = "conversations.json"):
        self.storage_file = storage_file
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.load_from_file()
    
    def load_from_file(self):
        """Load conversations from file if it exists"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.conversations = json.load(f)
                logger.info(f"📂 Loaded {len(self.conversations)} conversations from {self.storage_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load conversations from file: {e}")
            self.conversations = {}
    
    def save_to_file(self):
        """Save conversations to file"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 Saved conversations to {self.storage_file}")
        except Exception as e:
            logger.error(f"❌ Could not save conversations to file: {e}")
    
    def add_message(self, session_id: str, message_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to a conversation"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        message = {
            "id": f"{session_id}_{len(self.conversations[session_id])}",
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations[session_id].append(message)
        self.save_to_file()
        logger.debug(f"💬 Added {message_type} message to session {session_id}")
    
    def get_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for a session"""
        # Reload from file to get latest messages
        self.load_from_file()
        
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id]
        # Return most recent messages first (like database query)
        return list(reversed(messages[-limit:]))
    
    def get_conversation_summary(self, session_id: str, limit: int = 8) -> str:
        """Get a rolling summary of recent conversation"""
        messages = self.get_messages(session_id, limit)
        if not messages:
            return ""
        
        turn_texts = []
        for msg in messages:
            role = "Farmer" if msg["type"] == "user" else "AgriPal"
            content = msg["content"].replace("\n", " ")
            if len(content) > 160:
                content = content[:157] + "…"
            turn_texts.append(f"{role}: {content}")
        
        return " | ".join(turn_texts)
    
    def clear_session(self, session_id: str):
        """Clear all messages for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            self.save_to_file()
            logger.info(f"🗑️ Cleared conversation for session {session_id}")

# Global instance
conversation_storage = InMemoryConversationStorage()
