"""
🤖 AgriPal AI Agents Package
Collection of specialized AI agents for agricultural assistance using OpenAI Agents SDK.

This package contains three main agents:
- PerceptionAgent: Visual analysis of crop images using GPT-4o Vision
- KnowledgeAgent: RAG-powered agricultural knowledge retrieval with Weaviate + PostgreSQL
- EmailAgent: Intelligent report generation and email distribution

All agents are built using the OpenAI Agents SDK for enhanced capabilities and tool calling.
"""

from .perception_agent import PerceptionAgent
from .knowledge_agent import KnowledgeAgent
from .email_agent import EmailAgent

__all__ = ["PerceptionAgent", "KnowledgeAgent", "EmailAgent"]
