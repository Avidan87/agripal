"""
📚 Knowledge Agent - RAG-powered Agricultural Knowledge Retrieval
Specialized AI agent for searching and retrieving agricultural knowledge using 
Weaviate vector database, PostgreSQL context, and OpenAI Agents SDK.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Heavy imports - will be imported when needed via _import_heavy_dependencies()

# OpenAI Agents SDK imports
from openai import AsyncOpenAI
# OpenAI Agents SDK
from agents import Agent, Tool
from langchain.memory import ConversationBufferMemory

from ..models import (
    AgentMessage, 
    AgentResponse, 
    AgentType, 
    KnowledgeSearchResult
)
from ..config import settings, get_database_url

logger = logging.getLogger(__name__)

def _import_heavy_dependencies():
    """Import heavy dependencies only when needed"""
    try:
        import chromadb
        import asyncpg
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        
        # Return the imported modules as a dictionary
        return {
            'chromadb': chromadb,
            'asyncpg': asyncpg,
            'load_dataset': load_dataset,
            'HfApi': HfApi,
            'OpenAIEmbeddings': OpenAIEmbeddings,
            'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter,
            'Chroma': Chroma,
            'create_stuff_documents_chain': create_stuff_documents_chain,
            'ChatPromptTemplate': ChatPromptTemplate,
            'ChatOpenAI': ChatOpenAI
        }
    except ImportError as e:
        logger.warning(f"⚠️ Heavy dependencies not available: {e}")
        return None

class KnowledgeAgent:
    """
    📚 RAG-powered agricultural knowledge retrieval agent using OpenAI Agents SDK
    
    Capabilities:
    - Semantic search through agricultural knowledge base (Weaviate)
    - Contextual information retrieval from PostgreSQL
    - Weather data integration for localized recommendations
    - Agricultural best practices and research paper synthesis
    - Crop-specific and region-specific advice
    """
    
    def __init__(self):
        self.agent_type = AgentType.KNOWLEDGE
        self.model = settings.OPENAI_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Check if heavy dependencies are available
        self.dependencies = _import_heavy_dependencies()
        self.dependencies_available = self.dependencies is not None
        
        # Vector database setup
        self.chroma_client = None
        self.postgres_pool = None
        
        # Hugging Face setup
        self.hf_api = None
        self.hf_datasets_cache = {}
        
        # LangChain components
        self.langchain_embeddings = None
        self.langchain_vectorstore = None
        self.text_splitter = None
        self.retrieval_qa_chain = None
        self.conversation_memory = None
        
        # Knowledge base parameters
        self.max_search_results = 5
        self.similarity_threshold = 0.7
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        
        # Initialize OpenAI Agent SDK
        self.agent = None
        # Schedule async initialization immediately
        asyncio.create_task(self._initialize_async_components())
        
        logger.info("📚 Knowledge Agent initialized with OpenAI Agents SDK and LangChain")
    
    async def initialize(self):
        """
        Manual initialization method for testing and explicit setup
        """
        await self._initialize_async_components()
        return self
    
    async def _initialize_async_components(self):
        """
        Initialize async components (databases and agent)
        """
        try:
            # Initialize ChromaDB client
            await self._setup_chromadb()
            
            # Initialize PostgreSQL connection pool
            await self._setup_postgres()
            
            # Initialize Hugging Face client
            await self._setup_huggingface()
            
            # Initialize LangChain components
            await self._setup_langchain()
            
            # Preprocess and cache priority datasets
            if self.hf_api:
                # Check if ChromaDB needs population
                needs_population = await self._check_chromadb_population()
                if needs_population:
                    logger.info("📚 ChromaDB appears empty, populating with agricultural data...")
                    await self._preprocess_and_cache_datasets()
                else:
                    logger.info("📚 ChromaDB already populated with agricultural data")
            
            # Setup OpenAI Agent with tools
            await self._setup_agent()
            
            logger.info("✅ Knowledge Agent async components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Knowledge Agent async components: {str(e)}")
    
    async def _setup_chromadb(self):
        """
        Initialize ChromaDB vector database connection
        """
        try:
            # Initialize ChromaDB client with persistent storage
            if self.dependencies_available and 'chromadb' in self.dependencies:
                chromadb = self.dependencies['chromadb']
                self.chroma_client = chromadb.PersistentClient(
                    path=getattr(settings, 'CHROMA_DB_PATH', './chroma_db')
                )
            else:
                raise ImportError("chromadb not available")
            
            # Test connection by creating/getting a collection
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name="agricultural_knowledge",
                    metadata={"description": "Agricultural knowledge base for AgriPal"}
                )
                
                # Check if collection has the right embedding dimension
                if collection.count() > 0:
                    # Test embedding dimension by creating a test embedding
                    test_embedding = await self.client.embeddings.create(
                        input="test",
                        model=self.embedding_model
                    )
                    expected_dim = len(test_embedding.data[0].embedding)
                    
                    # Get a sample from the collection to check dimension
                    sample = collection.peek(limit=1)
                    if sample['embeddings'] and len(sample['embeddings'][0]) != expected_dim:
                        logger.warning(f"⚠️ Embedding dimension mismatch: collection has {len(sample['embeddings'][0])}, expected {expected_dim}")
                        logger.info("🔄 Recreating collection with correct embedding dimension...")
                        self.chroma_client.delete_collection("agricultural_knowledge")
                        collection = self.chroma_client.create_collection(
                            name="agricultural_knowledge",
                            metadata={"description": "Agricultural knowledge base for AgriPal"}
                        )
                        
            except Exception as e:
                logger.warning(f"⚠️ Collection setup issue: {str(e)}")
                # Try to create a fresh collection
                try:
                    self.chroma_client.delete_collection("agricultural_knowledge")
                except:
                    pass
                collection = self.chroma_client.create_collection(
                    name="agricultural_knowledge",
                    metadata={"description": "Agricultural knowledge base for AgriPal"}
                )
            
            logger.info("✅ ChromaDB connection established")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup ChromaDB: {str(e)}")
            self.chroma_client = None
    
    async def _setup_postgres(self):
        """
        Initialize PostgreSQL connection pool
        """
        try:
            # Try to connect with default settings
            if self.dependencies_available and 'asyncpg' in self.dependencies:
                asyncpg = self.dependencies['asyncpg']
                try:
                    # Convert SQLAlchemy URL to asyncpg format - use get_database_url() for Railway support
                    database_url = get_database_url()
                    asyncpg_url = database_url.replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")
                    
                    # Add SSL parameter to URL for Railway/Render PostgreSQL
                    if settings.ENVIRONMENT == "production" and ("railway" in asyncpg_url or "render.com" in asyncpg_url or "onrender.com" in asyncpg_url):
                        if "?" in asyncpg_url:
                            asyncpg_url += "&sslmode=require"
                        else:
                            asyncpg_url += "?sslmode=require"
                        logger.info("🔒 Using SSL=require for PostgreSQL connection")
                    
                    logger.info(f"🔗 Attempting PostgreSQL connection to: {asyncpg_url.split('@')[1] if '@' in asyncpg_url else 'hidden'}")
                    
                    # Configure SSL for Render PostgreSQL
                    ssl_config = None
                    if settings.ENVIRONMENT == "production":
                        # Render PostgreSQL requires SSL - use "require" not "prefer"
                        ssl_config = "require"  # Force SSL for Render
                        
                        # Additional SSL configuration for Render compatibility
                        if "render.com" in asyncpg_url or "onrender.com" in asyncpg_url:
                            # Render PostgreSQL requires SSL - keep "require"
                            ssl_config = "require"  # Render requires SSL, don't fallback
                    
                    # Additional SSL configuration for Render PostgreSQL - optimized for free plan
                    pool_kwargs = {
                        "min_size": 1,  # Reduced for free plan
                        "max_size": 3,  # Reduced for free plan
                        "max_inactive_connection_lifetime": 300,  # 5 minutes - Render free plan timeout
                        "command_timeout": 30,  # Command timeout
                        "server_settings": {
                            "jit": "off",
                            "application_name": "agripal_knowledge_agent",
                            "tcp_keepalives_idle": "300",  # Keep connections alive
                            "tcp_keepalives_interval": "30",
                            "tcp_keepalives_count": "3"
                        },
                        "ssl": ssl_config
                    }
                    
                    # Add SSL context for Render if needed
                    if ssl_config == "require" and ("render.com" in asyncpg_url or "onrender.com" in asyncpg_url):
                        # Try different SSL approaches for Render
                        try:
                            import ssl
                            ssl_context = ssl.create_default_context()
                            ssl_context.check_hostname = False
                            ssl_context.verify_mode = ssl.CERT_NONE
                            pool_kwargs["ssl"] = ssl_context
                            logger.info("🔒 Using SSL context for Render PostgreSQL connection")
                        except Exception as ssl_error:
                            logger.warning(f"⚠️ SSL context creation failed: {ssl_error}, using ssl=require")
                            pool_kwargs["ssl"] = "require"
                    elif ssl_config == "require":
                        logger.info("🔒 Using SSL=require for PostgreSQL connection")
                    
                    # Additional Render-specific SSL configuration
                    if ("render.com" in asyncpg_url or "onrender.com" in asyncpg_url):
                        # Ensure SSL is properly configured for Render
                        pool_kwargs.update({
                            "server_settings": {
                                "application_name": "agripal_knowledge_agent"
                            }
                        })
                    
                    # Try connection with comprehensive SSL configuration
                    try:
                        self.postgres_pool = await asyncpg.create_pool(
                            asyncpg_url,
                            **pool_kwargs
                        )
                        logger.info("✅ PostgreSQL connection pool created")
                    except Exception as ssl_error:
                        # If SSL context fails, try with simple ssl=require
                        if "ssl" in str(ssl_error).lower():
                            logger.warning(f"⚠️ SSL context failed, trying simple ssl=require: {ssl_error}")
                            pool_kwargs["ssl"] = "require"
                            self.postgres_pool = await asyncpg.create_pool(
                                asyncpg_url,
                                **pool_kwargs
                            )
                            logger.info("✅ PostgreSQL connection pool created with ssl=require")
                        else:
                            raise ssl_error
                except Exception as first_error:
                    # If that fails, try with the same credentials but different database name
                    fallback_url = get_database_url().replace("agripaldata", "agripal").replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")
                    
                    # Add SSL parameter to fallback URL for Railway/Render PostgreSQL
                    if settings.ENVIRONMENT == "production" and ("railway" in fallback_url or "render.com" in fallback_url or "onrender.com" in fallback_url):
                        if "?" in fallback_url:
                            fallback_url += "&sslmode=require"
                        else:
                            fallback_url += "?sslmode=require"
                    
                    logger.warning(f"⚠️ First PostgreSQL connection attempt failed: {str(first_error)}. Trying fallback...")
                    
                    try:
                        # Configure SSL for fallback connection too
                        ssl_config = None
                        if settings.ENVIRONMENT == "production":
                            ssl_config = "require"
                            
                            # Additional SSL configuration for Render compatibility
                            if "render.com" in fallback_url or "onrender.com" in fallback_url:
                                # Render PostgreSQL requires SSL - keep "require"
                                ssl_config = "require"  # Render requires SSL, don't fallback
                            
                        # Additional SSL configuration for fallback connection - optimized for free plan
                        pool_kwargs = {
                            "min_size": 1,  # Reduced for free plan
                            "max_size": 3,  # Reduced for free plan
                            "max_inactive_connection_lifetime": 300,  # 5 minutes - Render free plan timeout
                            "command_timeout": 30,  # Command timeout
                            "server_settings": {
                                "jit": "off",
                                "application_name": "agripal_knowledge_agent_fallback",
                                "tcp_keepalives_idle": "300",  # Keep connections alive
                                "tcp_keepalives_interval": "30",
                                "tcp_keepalives_count": "3"
                            },
                            "ssl": ssl_config
                        }
                        
                        # Add SSL context for Render if needed
                        if ssl_config == "require" and ("render.com" in fallback_url or "onrender.com" in fallback_url):
                            import ssl
                            ssl_context = ssl.create_default_context()
                            ssl_context.check_hostname = False
                            ssl_context.verify_mode = ssl.CERT_NONE
                            pool_kwargs["ssl"] = ssl_context
                            logger.info("🔒 Using SSL context for Render PostgreSQL fallback connection")
                        elif ssl_config == "require":
                            logger.info("🔒 Using SSL=require for PostgreSQL fallback connection")
                        
                        self.postgres_pool = await asyncpg.create_pool(
                            fallback_url,
                            **pool_kwargs
                        )
                        logger.info("✅ PostgreSQL connection pool created with fallback database")
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback PostgreSQL connection also failed: {str(fallback_error)}")
                        raise fallback_error
            else:
                raise ImportError("asyncpg not available")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup PostgreSQL: {str(e)}")
            self.postgres_pool = None
            # Don't raise the exception - allow the agent to work without PostgreSQL
            logger.warning("⚠️ Knowledge Agent will operate without PostgreSQL connection")
    
    async def _setup_huggingface(self):
        """
        Initialize Hugging Face API client and load agricultural datasets
        """
        try:
            if not self.dependencies_available:
                logger.warning("⚠️ Hugging Face dependencies not available")
                return
                
            # Get HfApi from dependencies
            HfApi = self.dependencies['HfApi']
            
            # Initialize Hugging Face API client
            self.hf_api = HfApi(token=getattr(settings, 'HF_TOKEN', None))
            
            # Note: We'll load datasets on-demand in the tool to avoid startup delays
            logger.info("✅ Hugging Face API client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Hugging Face: {str(e)}")
            self.hf_api = None
    
    async def _setup_langchain(self):
        """
        Initialize LangChain components for advanced knowledge processing
        """
        if not self.dependencies_available:
            logger.warning("⚠️ LangChain dependencies not available, skipping setup")
            return
            
        try:
            # Get dependencies from the imported modules
            OpenAIEmbeddings = self.dependencies['OpenAIEmbeddings']
            RecursiveCharacterTextSplitter = self.dependencies['RecursiveCharacterTextSplitter']
            Chroma = self.dependencies['Chroma']
            ChatOpenAI = self.dependencies['ChatOpenAI']
            ChatPromptTemplate = self.dependencies['ChatPromptTemplate']
            create_stuff_documents_chain = self.dependencies['create_stuff_documents_chain']
            
            # Initialize LangChain embeddings
            self.langchain_embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model=self.embedding_model
            )
            
            # Initialize text splitter for document processing
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize conversation memory for context retention
            self.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize LangChain ChromaDB vector store if ChromaDB is available
            if self.chroma_client:
                self.langchain_vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name="agricultural_knowledge",
                    embedding_function=self.langchain_embeddings
                )
                
                # Create retrieval QA chain for enhanced question answering
                llm = ChatOpenAI(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model_name=self.model,
                    temperature=0.1
                )
                
                # Define the system prompt
                system_prompt = (
                    "Use the given context to answer the agricultural question. "
                    "Provide evidence-based, practical recommendations that farmers can implement. "
                    "If you don't know the answer, say you don't know. "
                    "Context: {context}"
                )
                
                # Create the prompt template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                # Create the document combination chain
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                
                # Create the retrieval chain (temporarily disabled due to import issues)
                # self.retrieval_qa_chain = create_retrieval_chain(
                #     self.langchain_vectorstore.as_retriever(
                #         search_kwargs={"k": self.max_search_results}
                #     ),
                #     question_answer_chain
                # )
                self.retrieval_qa_chain = question_answer_chain
            
            logger.info("✅ LangChain components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup LangChain: {str(e)}")
            self.langchain_embeddings = None
            self.langchain_vectorstore = None
            self.retrieval_qa_chain = None
    
    async def _setup_agent(self):
        """
        Initialize the OpenAI Agent with RAG tools
        """
        try:
            # Define tools using dict-based schema to avoid SDK type instantiation issues
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_agricultural_knowledge",
                        "description": "Search the agricultural knowledge base for relevant information using semantic search",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query for agricultural knowledge"},
                                "crop_type": {"type": "string", "description": "Specific crop type to filter results"},
                                "region": {"type": "string", "description": "Geographic region for localized advice"},
                                "limit": {"type": "integer", "description": "Maximum number of results to return"}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_contextual_data",
                        "description": "Retrieve contextual information from user profiles and session history",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string", "description": "User ID for context retrieval"},
                                "session_id": {"type": "string", "description": "Session ID for history context"},
                                "crop_type": {"type": "string", "description": "Crop type for relevant history"}
                            },
                            "required": ["session_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_location_recommendations",
                        "description": "Get location-based agricultural recommendations using OpenCage Geocoding API",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "Location name for geocoding and recommendations"},
                                "crop_type": {"type": "string", "description": "Crop type for location-specific recommendations"},
                                "days_ahead": {"type": "integer", "description": "Number of days to consider for recommendations"}
                            },
                            "required": ["location"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "synthesize_recommendations",
                        "description": "Combine knowledge search results with context to create actionable recommendations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "search_results": {"type": "string", "description": "Results from knowledge search as JSON string"},
                                "context_data": {"type": "string", "description": "Contextual user and session data as JSON string"},
                                "weather_data": {"type": "string", "description": "Weather information as JSON string"},
                                "user_query": {"type": "string", "description": "Original user question"}
                            },
                            "required": ["search_results", "user_query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_huggingface_data",
                        "description": "Fetch agricultural data and research from Hugging Face datasets and models",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query for agricultural information"},
                                "dataset_name": {"type": "string", "description": "Specific Hugging Face dataset to search (optional)"},
                                "crop_type": {"type": "string", "description": "Crop type for filtering relevant information"},
                                "limit": {"type": "integer", "description": "Maximum number of results to return"}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "populate_knowledge_base",
                        "description": "Populate ChromaDB with Hugging Face agricultural datasets for enhanced knowledge retrieval",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "max_documents_per_dataset": {"type": "integer", "description": "Maximum number of documents to process per dataset"},
                                "force_rebuild": {"type": "boolean", "description": "Whether to rebuild the knowledge base from scratch"}
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "langchain_enhanced_search",
                        "description": "Enhanced knowledge search using LangChain with advanced document processing and retrieval",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query for agricultural knowledge"},
                                "search_type": {"type": "string", "description": "Type of search to perform"},
                                "crop_type": {"type": "string", "description": "Crop type for filtering results"},
                                "include_sources": {"type": "boolean", "description": "Whether to include source documents"}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "manage_hf_datasets",
                        "description": "Manage Hugging Face datasets - cache, preprocess, and get dataset statistics",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "description": "Action to perform on datasets"},
                                "dataset_names": {"type": "string", "description": "Optional specific dataset names to manage as comma-separated string"},
                                "priority_level": {"type": "string", "description": "Priority level for caching"}
                            },
                            "required": ["action"]
                        }
                    }
                }
            ]

            # Create the agent with agricultural knowledge expertise
            self.agent = Agent(
                name="AgriPal Knowledge Agent",
                instructions="""Expert agricultural knowledge retrieval and synthesis agent with access to comprehensive farming databases.

You are a supportive, empathetic agricultural specialist. Communicate clearly, be encouraging, and avoid alarming language unless risk is high. Use a friendly, professional tone.
Add a touch of warmth with tasteful emojis (1–2) when appropriate; avoid overuse.

You have access to:
- Extensive agricultural knowledge base (FAO manuals, research papers, extension guides)
- Hugging Face datasets and models for cutting-edge agricultural research
- Historical user data and farming context
- OpenCage Geocoding API for precise location services
- Real-time weather data and forecasts via OpenWeatherMap API
- Crop-specific and region-specific best practices

Always provide:
- Evidence-based guidance with source awareness
- Context-aware recommendations considering local conditions and weather
- Practical, step-by-step actions
- Sensible reassurance for minor issues; clear, calm guidance for severe cases
- Friendly phrasing with light emoji use where it enhances clarity and engagement
""",
                tools=tools
            )

            # Create tool mapping for proper function execution
            self.tool_mapping = {
                "search_agricultural_knowledge": self._tool_search_knowledge,
                "get_contextual_data": self._tool_get_contextual_data,
                "get_location_recommendations": self._tool_get_weather_recommendations,
                "synthesize_recommendations": self._tool_synthesize_recommendations,
                "fetch_huggingface_data": self._tool_fetch_huggingface_data,
                "populate_knowledge_base": self._tool_populate_knowledge_base,
                "langchain_enhanced_search": self._tool_langchain_enhanced_search,
                "manage_hf_datasets": self._tool_manage_hf_datasets
            }

            logger.info("✅ Knowledge Agent successfully initialized with RAG tools and tool mapping")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Knowledge Agent: {str(e)}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            self.agent = None
    
    async def _fallback_direct_tool_calls(self, user_query: str, crop_type: str, user_context: Dict[str, Any], perception_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback method for direct tool calls when agent.run is not available
        """
        try:
            # Enhance search query with perception context if available
            enhanced_query = user_query
            if perception_results:
                image_analysis = perception_results.get('image_analysis', {})
                issues = image_analysis.get('detected_issues', [])
                if issues:
                    # Add perception context to the search query
                    perception_terms = " ".join(issues[:3])  # Use top 3 issues
                    enhanced_query = f"{user_query} {perception_terms}"
                    logger.info(f"🔍 Enhanced search query with perception context: {enhanced_query}")
            
            # Use direct tool call for knowledge search
            search_result = await self._tool_search_knowledge(
                query=enhanced_query,
                crop_type=crop_type,
                region=user_context.get('location', ''),
                limit=self.max_search_results
            )
            
            # Parse the JSON result
            search_data = json.loads(search_result)
            
            # Optionally enrich with weather context when location is available
            weather_block = None
            user_location_for_weather = user_context.get('location') if user_context else None
            if user_location_for_weather and isinstance(user_location_for_weather, str) and user_location_for_weather.strip():
                try:
                    weather_json = await self._tool_get_weather_recommendations(
                        location=user_location_for_weather.strip(),
                        crop_type=crop_type,
                        days_ahead=7
                    )
                    try:
                        weather_block = json.loads(weather_json) if isinstance(weather_json, str) else weather_json
                    except json.JSONDecodeError:
                        weather_block = {"error": "Invalid weather data"}
                except Exception as wx_err:
                    logger.warning(f"⚠️ Weather enrichment failed: {wx_err}")

            # Create a structured response
            documents = search_data.get("documents", [])
            if documents:
                # If we have documents, create a proper response
                content = f"Based on my search, I found {len(documents)} relevant sources about {user_query}. "
                if crop_type:
                    content += f"Here's what I found about {crop_type}: "
                
                # Add key information from the documents
                for i, doc in enumerate(documents[:3]):  # Show top 3 results
                    content += f"\n\n{i+1}. {doc.get('content', '')[:200]}..."
            else:
                # No documents found - check if we should use perception results
                query_lower = user_query.lower()
                image_related_keywords = ["image", "photo", "picture", "see", "show", "look", "visual", "appears", "looks like", "what is wrong", "diagnose", "identify"]
                is_image_related_query = any(keyword in query_lower for keyword in image_related_keywords)
                
                if perception_results and is_image_related_query:
                    image_analysis = perception_results.get('image_analysis', {})
                    analysis_text = image_analysis.get('analysis_text', '')
                    
                    if analysis_text and analysis_text.strip():
                        # Use the perception agent's analysis as the main response
                        content = analysis_text.strip()
                        logger.info("🔍 Using perception analysis as primary response")
                    else:
                        # Fallback to structured data if no analysis_text
                        issues = image_analysis.get('detected_issues', [])
                        health_score = image_analysis.get('crop_health_score', 'Unknown')
                        severity = image_analysis.get('severity', 'Unknown')
                        recommendations = image_analysis.get('recommendations', [])
                        
                        content = f"Based on the visual analysis of your crop image, "
                        if issues and issues != ["Unable to determine specific issues from image"]:
                            content += f"I can see signs of {', '.join(issues[:2])}. "
                        content += f"The overall health score appears to be {health_score} with {severity} severity. "
                        
                        if recommendations:
                            content += "\n\nHere are my specific recommendations:\n"
                            for i, rec in enumerate(recommendations[:3], 1):
                                content += f"{i}. {rec}\n"
                        else:
                            content += "I recommend monitoring your crop closely and consulting with local agricultural experts for specific treatment options."
                        
                        logger.info("🔍 Using structured perception data as response")
                else:
                    # No documents found - use OpenAI's general agricultural knowledge
                    logger.info("🔍 No documents found, using general agricultural knowledge")
                    content = await self._generate_general_agricultural_response(user_query, crop_type, user_context)
                
                # Only add generic advice if the response is very short (likely a fallback)
                if len(content) < 200:
                    content += "\n\nAdditional farming tips:"
                    if crop_type:
                        content += f"\n• For {crop_type} crops, ensure proper spacing and drainage"
                        content += f"\n• Monitor {crop_type} for common pests and diseases in your region"
                    content += "\n• Regular soil testing helps identify nutrient needs"
                    content += "\n• Maintain consistent watering schedule based on crop requirements"
            
            # Append concise weather summary if available
            if weather_block:
                combined_recs = weather_block.get("combined_recommendations") or weather_block.get("weather_recommendations") or []
                if combined_recs:
                    content += "\n\nConsidering the weather for your location:"
                    for rec in combined_recs[:3]:
                        content += f"\n• {rec}"
            
            return {
                "content": content,
                "sources": documents,
                "metadata": {
                    "query": user_query,
                    "crop_type": crop_type,
                    "search_results_count": len(documents),
                    "weather_included": bool(weather_block)
                }
            }
            
        except Exception as tool_error:
            logger.warning(f"⚠️ Direct tool call failed: {tool_error}")
            # Fallback to basic response
            return {
                "content": f"Knowledge search for '{user_query}' is temporarily unavailable. Please try again later.",
                "sources": [],
                "metadata": {"error": str(tool_error)}
            }
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        """
        Main processing method for knowledge agent using OpenAI Agents SDK
        
        Args:
            message: AgentMessage containing search query and context
            
        Returns:
            AgentResponse with knowledge search results and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"📚 Processing knowledge request for session {message.session_id}")
            
            # If agent isn't initialized, try to initialize it now
            if not self.agent:
                logger.warning("⚠️ Agent not initialized yet, attempting initialization now")
                await self._setup_agent()
                
            # Extract query and context first to avoid variable reference issues
            user_query = message.content.get('query', '')
            crop_type = message.content.get('crop_type')
            
            # If still not initialized, use direct search instead
            if not self.agent:
                logger.warning("⚠️ Agent initialization failed, falling back to direct search")
                # Use direct search method instead
                search_results = await self._tool_search_knowledge(user_query, crop_type)
                
                # Create a simple response
                knowledge_result = KnowledgeSearchResult(
                    relevant_documents=[],
                    search_query=user_query,
                    confidence_scores=[],
                    source_types=[],
                    contextual_advice="Using direct search due to agent initialization failure."
                )
                
                processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    session_id=message.session_id,
                    success=True,
                    result=knowledge_result.dict(),
                    processing_time_ms=processing_time
                )
            
            # Extract additional context
            user_context = message.content.get('user_context', {})
            perception_results = message.content.get('perception_context')
            
            if not user_query:
                raise ValueError("No query provided for knowledge search")
            
            # Weather intent detection and handling
            if self._is_weather_intent(user_query):
                user_location = (user_context or {}).get('location') if 'user_context' in message.content else None
                if user_location and isinstance(user_location, str) and user_location.strip():
                    try:
                        weather_json = await self._tool_get_weather_recommendations(
                            location=user_location.strip(),
                            crop_type=crop_type,
                            days_ahead=7
                        )
                        try:
                            weather_data = json.loads(weather_json) if isinstance(weather_json, str) else weather_json
                        except json.JSONDecodeError:
                            weather_data = {"error": "Invalid weather data"}

                        combined_recs = weather_data.get("combined_recommendations") or weather_data.get("weather_recommendations") or []
                        advice_lines = []
                        if combined_recs:
                            advice_lines.append(f"Weather outlook for {user_location}:")
                            advice_lines.extend([str(r) for r in combined_recs[:5]])
                        else:
                            advice_lines.append(f"Unable to retrieve detailed weather recommendations for {user_location} right now. Please try again later.")

                        knowledge_result = KnowledgeSearchResult(
                            relevant_documents=[],
                            search_query=user_query,
                            confidence_scores=[],
                            source_types=[],
                            contextual_advice="\n".join(advice_lines)
                        )

                        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                        # Include raw weather block alongside standard fields for UI extensions
                        result_dict = knowledge_result.dict()
                        result_dict["weather"] = weather_data

                        return AgentResponse(
                            agent_type=self.agent_type,
                            session_id=message.session_id,
                            success=True,
                            result=result_dict,
                            processing_time_ms=processing_time
                        )
                    except Exception as weather_err:
                        logger.warning(f"⚠️ Weather flow failed, falling back to knowledge search: {weather_err}")
                # If no usable location or weather failed, fall through to normal knowledge search

            # Create agent prompt with context
            agent_prompt = self._build_agent_prompt(
                user_query, 
                crop_type, 
                user_context, 
                perception_results
            )
            
            # Use proper agent execution with tool mapping
            if self.agent and hasattr(self.agent, "run"):
                try:
                    # Use agent.run with proper tool context
                    run_result = await self.agent.run(
                        message=agent_prompt,
                        tools_context={
                            "user_query": user_query,
                            "crop_type": crop_type,
                            "user_context": user_context,
                            "perception_results": perception_results,
                            "session_id": message.session_id
                        }
                    )
                    
                    # Parse agent response into structured format
                    if hasattr(run_result, 'content'):
                        # Handle agent response object
                        content = run_result.content
                        sources = getattr(run_result, 'sources', [])
                        metadata = getattr(run_result, 'metadata', {})
                    else:
                        # Handle dict response
                        content = run_result.get('content', '')
                        sources = run_result.get('sources', [])
                        metadata = run_result.get('metadata', {})
                    
                    run_result = {
                        "content": content,
                        "sources": sources,
                        "metadata": metadata
                    }
                    
                except Exception as agent_error:
                    logger.warning(f"⚠️ Agent execution failed: {agent_error}, falling back to direct tool calls")
                    # Fallback to direct tool calls
                    run_result = await self._fallback_direct_tool_calls(
                        user_query, crop_type, user_context, perception_results
                    )
            else:
                # No agent available or agent.run not supported, use direct tool calls
                logger.info("ℹ️ Using direct tool calls (agent.run not available)")
                run_result = await self._fallback_direct_tool_calls(
                    user_query, crop_type, user_context, perception_results
                )
            
            # Parse agent response into structured format
            knowledge_result = await self._parse_agent_response(run_result, user_query)
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # If we computed weather earlier in the flow, attach it for downstream consumers
            result_payload = knowledge_result.dict()
            try:
                if 'metadata' in run_result and run_result['metadata'].get('weather_included'):
                    # Attach a minimal weather flag; full blocks are returned in the weather-intent branch above
                    result_payload.setdefault('extras', {})['weather_context_used'] = True
            except Exception:
                pass

            return AgentResponse(
                agent_type=self.agent_type,
                session_id=message.session_id,
                success=True,
                result=result_payload,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"❌ Knowledge search failed: {str(e)}")
            
            return AgentResponse(
                agent_type=self.agent_type,
                session_id=message.session_id,
                success=False,
                result={},
                error=str(e),
                processing_time_ms=processing_time
            )

    def _is_weather_intent(self, text: str) -> bool:
        """
        Simple heuristic to detect weather-related intent in a user query.
        """
        if not text:
            return False
        lowered = text.lower()
        keywords = [
            "weather", "forecast", "temperature", "rain", "rainfall", "precip",
            "humidity", "wind", "uv", "sunlight", "frost", "monsoon", "storm"
        ]
        return any(k in lowered for k in keywords)
    
    def _build_agent_prompt(
        self, 
        user_query: str, 
        crop_type: Optional[str], 
        user_context: Dict[str, Any],
        perception_results: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build agent prompt for OpenAI Agents SDK
        
        Args:
            user_query: User's agricultural question
            crop_type: Type of crop if known
            user_context: User profile and context
            perception_results: Results from perception agent if available
            
        Returns:
            Formatted prompt for the agent
        """
        crop_context = f"Crop: {crop_type}. " if crop_type else ""
        location_context = f"Location: {user_context.get('location', 'Unknown')}. " if user_context.get('location') else ""
        
        perception_context = ""
        if perception_results:
            image_analysis = perception_results.get('image_analysis', {})
            issues = image_analysis.get('detected_issues', [])
            top_issue = (issues[0] if isinstance(issues, list) and issues else None) or "Unknown"
            health_score = image_analysis.get('crop_health_score', 'Unknown')
            severity = image_analysis.get('severity', 'Unknown')
            confidence = image_analysis.get('confidence_level', 'Unknown')
            observations = (
                image_analysis.get('metadata', {}).get('observations')
                or image_analysis.get('observations') or ""
            )
            # Provide rich but compact perception cues without raw formatting headers
            perception_context = (
                f"Visual analysis cues → likely: {top_issue}; severity: {severity}; "
                f"confidence: {confidence}; key observations: {observations}. "
            )
        
        # Extract conversation context if available
        conversation_context = ""
        if user_context:
            rolling_summary = user_context.get('rolling_summary', '')
            agricultural_context = user_context.get('agricultural_context', {})
            
            if rolling_summary:
                conversation_context += f"Recent conversation: {rolling_summary}. "
            
            if agricultural_context:
                crops_discussed = ", ".join(agricultural_context.get("crops", []))
                problems_discussed = ", ".join(agricultural_context.get("problems", []))
                if crops_discussed or problems_discussed:
                    conversation_context += f"Previous topics: crops ({crops_discussed or 'none'}), issues ({problems_discussed or 'none'}). "
        
        return f"""
        You are AgriPal, an experienced agricultural consultant helping a farmer with their ongoing question.
        
        FARMER'S QUESTION: {user_query}
        
        FARM CONTEXT:
        {crop_context}{location_context}{perception_context}
        
        CONVERSATION HISTORY:
        {conversation_context}
        
        YOUR ROLE:
        Provide strategic, engaging, and practical agricultural guidance. Be conversational and insightful, 
        as if you're continuing an ongoing conversation with a farmer you know well. Focus on meaningful insights 
        that help them make informed decisions.
        
        RESPONSE APPROACH:
        - Write naturally, as if continuing an ongoing conversation - no need for greetings like "Hey there!"
        - Reference previous conversations when relevant (e.g., "Following up on the corn issue...", "As we discussed...")
        - Show continuity and memory of their farming journey
        - Provide strategic insights that go beyond basic information
        - Be engaging and show genuine interest in their farming progress
        - Use evidence to support your insights, but don't expose technical details
        - Adapt your tone based on whether this continues a previous topic or starts a new one
        - Offer practical wisdom and real-world context when relevant
        - Be encouraging and supportive, especially for ongoing challenges
        - Use relevant emojis throughout your response to make it more engaging and visually appealing
        - Include 3-5 emojis per response that relate to the content (🌾🌱🌿💧☀️🌧️🐛🔬📊✅❌⚠️💡🎯🚀🌿🌽🍅🥕🌶️🥬🌾🌻🌺🌸🌼🌷)
        
        TOOLS TO USE:
        1. Search the agricultural knowledge base for relevant information
        2. Fetch additional research data from trusted datasets
        3. Consider local climate and regional conditions for their location
        4. Synthesize all information into strategic, actionable advice
        
        Create a response that:
        - Directly addresses their question with strategic insight
        - References previous conversations naturally when appropriate
        - Provides meaningful context and practical guidance
        - Engages them as a continuing conversation
        - Shows understanding of their farming context and history
        - Offers actionable next steps when appropriate
        
        Write your response naturally, as if you're continuing your ongoing relationship with this farmer.
        """
    
    async def _check_chromadb_population(self) -> bool:
        """
        Check if ChromaDB needs to be populated with agricultural data
        
        Returns:
            True if population is needed, False if already populated
        """
        try:
            if not self.chroma_client:
                logger.warning("⚠️ ChromaDB client not available")
                return False
                
            collection = self.chroma_client.get_collection("agricultural_knowledge")
            doc_count = collection.count()
            
            # Consider it populated if we have at least 50 documents
            # (Should be around 200 from the two datasets: Mahesh2841/Agriculture + Dharine/agriculture-10k)
            is_populated = doc_count >= 50
            
            logger.info(f"📊 ChromaDB has {doc_count} documents, populated: {is_populated}")
            return not is_populated
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to check ChromaDB population: {str(e)}")
            return True  # Assume needs population if we can't check

    async def _populate_chromadb_with_docs(self, documents: List[Dict[str, Any]], dataset_name: str) -> None:
        """
        Populate ChromaDB with processed agricultural documents and their embeddings
        """
        try:
            if not self.chroma_client:
                logger.warning("⚠️ ChromaDB client not available for population")
                return
                
            collection = self.chroma_client.get_collection("agricultural_knowledge")
            
            # Prepare documents for ChromaDB
            doc_ids = []
            doc_texts = []
            doc_embeddings = []
            doc_metadatas = []
            
            logger.info(f"📚 Populating ChromaDB with {len(documents)} documents from {dataset_name}")
            
            for i, doc in enumerate(documents):
                try:
                    # Generate unique ID
                    doc_id = f"{dataset_name}_{i}_{hash(doc.get('content', ''))}"
                    doc_ids.append(doc_id)
                    
                    # Extract text content
                    content = doc.get('content', '')
                    if len(content) > 8000:  # Limit content length for embeddings
                        content = content[:8000] + "..."
                    doc_texts.append(content)
                    
                    # Generate embedding
                    embedding_response = await self.client.embeddings.create(
                        input=content,
                        model=self.embedding_model
                    )
                    doc_embeddings.append(embedding_response.data[0].embedding)
                    
                    # Prepare metadata
                    metadata = {
                        "dataset": dataset_name,
                        "title": doc.get('title', ''),
                        "crop_type": doc.get('crop_type', ''),
                        "category": doc.get('category', 'general'),
                        "source": doc.get('source', 'huggingface')
                    }
                    doc_metadatas.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process document {i}: {str(e)}")
                    continue
            
            # Add to ChromaDB in batch
            if doc_ids:
                collection.add(
                    ids=doc_ids,
                    documents=doc_texts,
                    embeddings=doc_embeddings,
                    metadatas=doc_metadatas
                )
                logger.info(f"✅ Added {len(doc_ids)} documents to ChromaDB from {dataset_name}")
            else:
                logger.warning(f"⚠️ No valid documents to add from {dataset_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to populate ChromaDB with {dataset_name}: {str(e)}")

    async def _generate_general_agricultural_response(self, user_query: str, crop_type: Optional[str], user_context: Dict[str, Any]) -> str:
        """
        Generate helpful agricultural response using OpenAI's general knowledge when knowledge base is empty
        """
        try:
            # Build a prompt that leverages OpenAI's agricultural knowledge
            system_prompt = """You are an experienced agricultural consultant with deep knowledge of crop production, pest management, plant diseases, and sustainable farming practices. 

Provide practical, actionable advice for farmers. Be conversational and engaging, as if speaking directly to a farmer who needs help. Use relevant emojis (🌾🌱🐛💧🌧️☀️🌿💡) to make responses more engaging.

For pest-related questions, include:
- What damage the pest causes
- How to identify the problem
- Prevention strategies
- Treatment options (both organic and conventional)
- Long-term management

For crop diseases, include:
- Symptoms to look for
- Conditions that favor the disease
- Prevention measures
- Treatment options

Always be encouraging and supportive, helping farmers feel confident about managing their crops."""

            user_prompt = f"""Question: {user_query}

Context: 
- Crop type: {crop_type or 'not specified'}
- Location: {user_context.get('location', 'not specified')}

Please provide a comprehensive, helpful response that directly answers this farmer's question. Be specific and actionable."""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error("Failed to generate general agricultural response: %s", str(e))
            # Enhanced fallback with specific agricultural guidance
            query_lower = user_query.lower()
            
            if any(keyword in query_lower for keyword in ['aphid', 'pest', 'insect', 'bug']):
                if 'potato' in query_lower:
                    return """Aphids can cause significant damage to potato crops! 🐛🥔 These small, soft-bodied insects feed on plant sap and can:

• Stunt plant growth by draining nutrients
• Transmit viral diseases (like potato virus Y and potato leafroll virus)
• Cause yellowing and curling of leaves
• Reduce tuber yield and quality
• Create honeydew that leads to sooty mold

Prevention and control options:
🌱 Encourage beneficial insects like ladybugs and lacewings
💧 Use reflective mulches to confuse aphids
🌿 Apply neem oil or insecticidal soap for organic control
🔬 Consider systemic insecticides for severe infestations
🛡️ Monitor regularly, especially during warm weather

Early detection is key - check the undersides of leaves regularly for small clusters of these pests!"""
                else:
                    return """Aphids are common agricultural pests that can damage many crops! 🐛 These small insects feed on plant sap and can cause stunting, yellowing, and transmit plant viruses. Regular monitoring and integrated pest management approaches work best for control."""
            
            elif any(keyword in query_lower for keyword in ['disease', 'fungal', 'bacterial', 'blight']):
                return "Plant diseases can significantly impact crop yields. Early identification and proper management are crucial. Consider factors like humidity, temperature, and plant stress when developing treatment strategies. 🌿🔬"
            
            elif any(keyword in query_lower for keyword in ['nutrient', 'fertilizer', 'deficiency']):
                return "Proper nutrition is essential for healthy crop development. Soil testing can help identify specific nutrient needs. Consider both macro and micronutrients for optimal plant health. 🌱💡"
            
            else:
                return f"I'd be happy to help with your question about {user_query}. Could you provide more specific details about what you're observing with your crops? This will help me give you more targeted advice. 🌾"

    async def _parse_agent_response(self, run_result: Any, original_query: str) -> KnowledgeSearchResult:
        """
        Parse OpenAI Agent response into structured KnowledgeSearchResult
        
        Args:
            run_result: Result from agent run
            original_query: Original search query
            
        Returns:
            Structured KnowledgeSearchResult
        """
        try:
            # Extract response from agent run
            if hasattr(run_result, 'response'):
                response_text = run_result.response
            elif isinstance(run_result, dict):
                # Handle mock response structure
                response_text = run_result.get('content', 'No information found')
                relevant_documents = run_result.get('sources', [])
                confidence_scores = []
                source_types = []
                contextual_advice = response_text
            else:
                response_text = str(run_result)
                relevant_documents = []
                confidence_scores = []
                source_types = []
                contextual_advice = response_text
            
            # Check if agent used tools and extract tool results
            if hasattr(run_result, 'tool_calls') and run_result.tool_calls:
                for tool_call in run_result.tool_calls:
                    if tool_call.function.name == "search_agricultural_knowledge":
                        tool_result = json.loads(tool_call.function.result) if tool_call.function.result else {}
                        relevant_documents.extend(tool_result.get('documents', []))
                        confidence_scores.extend(tool_result.get('confidence_scores', []))
                        source_types.extend(tool_result.get('source_types', []))
                    
                    elif tool_call.function.name == "synthesize_recommendations":
                        synthesis_result = json.loads(tool_call.function.result) if tool_call.function.result else {}
                        contextual_advice = synthesis_result.get('recommendations', response_text)
            elif isinstance(run_result, dict) and 'sources' in run_result:
                # We already processed the mock response above, no need to do it again
                pass
            
            return KnowledgeSearchResult(
                relevant_documents=relevant_documents,
                search_query=original_query,
                confidence_scores=confidence_scores,
                source_types=source_types,
                contextual_advice=contextual_advice
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse knowledge agent response: {str(e)}")
            
            # Provide helpful fallback advice for error case
            # Note: message variable not available in this context, using empty dict
            user_context = {}
            
            fallback_advice = "I encountered a technical issue retrieving specific knowledge. Try rephrasing your question with more specific details about your crop, symptoms, or farming situation. I'm here to help with agricultural guidance!"
            
            return KnowledgeSearchResult(
                relevant_documents=[],
                search_query=original_query,
                confidence_scores=[],
                source_types=[],
                contextual_advice=fallback_advice
            )
    
    # Tool functions for OpenAI Agents SDK
    async def _tool_search_knowledge(
        self, 
        query: str, 
        crop_type: Optional[str] = None, 
        region: Optional[str] = None, 
        limit: int = 5
    ) -> str:
        """
        Tool function for searching agricultural knowledge using Weaviate
        
        Args:
            query: Search query
            crop_type: Optional crop type filter
            region: Optional region filter
            limit: Maximum results to return
            
        Returns:
            JSON string with search results
        """
        try:
            if not self.chroma_client:
                return json.dumps({
                    "error": "ChromaDB client not available",
                    "documents": [],
                    "confidence_scores": [],
                    "source_types": []
                })
            
            # Get the collection
            collection = self.chroma_client.get_collection("agricultural_knowledge")
            
            # Build where clause for filtering - ChromaDB requires simpler syntax
            where_clause = None
            if crop_type and region:
                # If both are provided, use crop_type as primary filter
                where_clause = {"crop_type": crop_type}
            elif crop_type:
                where_clause = {"crop_type": crop_type}
            elif region:
                # For region, we'll search without where clause and filter results
                where_clause = None
            
            # Perform semantic search using ChromaDB
            results = collection.query(
                query_texts=[query],
                n_results=limit * 2,  # Get more results to filter by region if needed
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract results
            processed_docs = []
            confidence_scores = []
            source_types = []
            
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 1.0
                    
                    # Filter by region if specified and not already filtered by where clause
                    doc_region = metadata.get("region", "global")
                    if region and where_clause is None:
                        # Only include if region matches or is global
                        if doc_region not in [region, "global"]:
                            continue
                    
                    processed_docs.append({
                        "content": doc,
                        "source": metadata.get("source", "Unknown"),
                        "crop_type": metadata.get("crop_type", "general"),
                        "region": doc_region,
                        "document_type": metadata.get("document_type", "unknown")
                    })
                    
                    # Convert distance to confidence (lower distance = higher confidence)
                    confidence = max(0.0, 1.0 - distance)
                    confidence_scores.append(confidence)
                    
                    source_types.append(metadata.get("document_type", "unknown"))
                    
                    # Limit results to requested amount
                    if len(processed_docs) >= limit:
                        break
            
            return json.dumps({
                "documents": processed_docs,
                "confidence_scores": confidence_scores,
                "source_types": source_types,
                "total_results": len(processed_docs)
            })
            
        except Exception as e:
            logger.error(f"❌ Knowledge search failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "documents": [],
                "confidence_scores": [],
                "source_types": []
            })
    
    async def _tool_get_contextual_data(
        self, 
        session_id: str, 
        user_id: Optional[str] = None, 
        crop_type: Optional[str] = None
    ) -> str:
        """
        Tool function for retrieving contextual data from PostgreSQL
        
        Args:
            session_id: Session ID for context
            user_id: Optional user ID
            crop_type: Optional crop type for history filtering
            
        Returns:
            JSON string with contextual data
        """
        try:
            if not self.postgres_pool:
                return json.dumps({
                    "error": "PostgreSQL pool not available",
                    "context": {}
                })
            
            async with self.postgres_pool.acquire() as conn:
                # Get session data
                session_data = await conn.fetchrow("""
                    SELECT cs.*, u.email, u.name, u.location, u.farm_details
                    FROM chat_sessions cs
                    LEFT JOIN users u ON cs.user_id = u.id
                    WHERE cs.id = $1
                """, session_id)
                
                # Get recent similar sessions if user_id available
                session_history = []
                if user_id and crop_type:
                    history_records = await conn.fetch("""
                        SELECT cs.summary, ar.recommendations, cs.created_at
                        FROM chat_sessions cs
                        JOIN analysis_reports ar ON cs.id = ar.session_id
                        WHERE cs.user_id = $1 
                        AND ar.analysis_results->>'crop_type' = $2
                        ORDER BY cs.created_at DESC LIMIT 3
                    """, user_id, crop_type)
                    
                    session_history = [dict(record) for record in history_records]
                
                context = {
                    "session_data": dict(session_data) if session_data else {},
                    "session_history": session_history,
                    "user_profile": {
                        "location": session_data["location"] if session_data else None,
                        "farm_details": session_data["farm_details"] if session_data else None
                    }
                }
                
                return json.dumps(context, default=str)
                
        except Exception as e:
            logger.error(f"❌ Contextual data retrieval failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "context": {}
            })
    
    async def _tool_get_weather_recommendations(
        self, 
        location: str, 
        crop_type: Optional[str] = None, 
        days_ahead: int = 7
    ) -> str:
        """
        Tool function for getting location-based agricultural recommendations using Amazon Location Services
        
        Args:
            location: Location name for geocoding and recommendations
            crop_type: Optional crop type for specific recommendations
            days_ahead: Number of days to consider for recommendations
            
        Returns:
            JSON string with location-based agricultural recommendations
        """
        try:
            # Use OpenCage Geocoding API for geocoding
            coordinates = await self._geocode_location_with_opencage(location)
            
            if not coordinates:
                return json.dumps({
                    "error": "Unable to geocode location",
                    "recommendations": [
                        "Please verify the location name and try again",
                        "Consider using a more specific location (city, state/province, country)"
                    ]
                })
            
            lat, lon = coordinates
            
            # Get current weather data using the coordinates
            weather_data = await self._get_weather_data(lat, lon, days_ahead)
            
            # Generate location-based agricultural recommendations
            regional_recommendations = await self._generate_regional_recommendations(
                lat, lon, location, crop_type
            )
            
            # Generate weather-based agricultural recommendations
            weather_recommendations = await self._generate_weather_recommendations(
                weather_data, crop_type
            )
            
            # Combine all recommendations
            all_recommendations = regional_recommendations + weather_recommendations
            
            return json.dumps({
                "coordinates": {"latitude": lat, "longitude": lon},
                "location_verified": location,
                "weather_data": weather_data,
                "regional_recommendations": regional_recommendations,
                "weather_recommendations": weather_recommendations,
                "combined_recommendations": all_recommendations[:10],  # Top 10 combined
                "geocoding_service": "opencage_geocoding",
                "weather_service": "openweathermap",
                "crop_specific_advice": f"For {crop_type}: Monitor local growing conditions and seasonal patterns" if crop_type else None
            })
            
        except Exception as e:
            logger.error(f"❌ Location-based recommendations failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "recommendations": ["Unable to retrieve location-based recommendations"]
            })
    
    async def _tool_synthesize_recommendations(
        self, 
        search_results: str, 
        user_query: str,
        context_data: Optional[str] = None, 
        weather_data: Optional[str] = None
    ) -> str:
        """
        Tool function for synthesizing comprehensive recommendations
        
        Args:
            search_results: Results from knowledge search
            user_query: Original user question
            context_data: Optional user context
            weather_data: Optional weather information
            
        Returns:
            JSON string with synthesized recommendations
        """
        try:
            # Parse search results from JSON string
            try:
                search_results_list = json.loads(search_results) if isinstance(search_results, str) else search_results
            except (json.JSONDecodeError, TypeError):
                search_results_list = []
            
            # Extract key information from search results
            sources = []
            key_points = []
            
            for result in search_results_list:
                if isinstance(result, dict):
                    sources.append(result.get('source', 'Unknown'))
                    key_points.append(result.get('content', '')[:200] + "...")
            
            # Build comprehensive recommendations
            recommendations = []
            
            # Add knowledge-based recommendations
            if key_points:
                recommendations.append("Based on agricultural research and best practices:")
                recommendations.extend([f"• {point}" for point in key_points[:3]])
            
            # Add weather-based recommendations
            if weather_data:
                try:
                    weather_dict = json.loads(weather_data) if isinstance(weather_data, str) else weather_data
                    weather_recs = weather_dict.get('recommendations', [])
                    if weather_recs:
                        recommendations.append("Considering current weather conditions:")
                        recommendations.extend([f"• {rec}" for rec in weather_recs[:2]])
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # Add context-based recommendations
            if context_data:
                try:
                    context_dict = json.loads(context_data) if isinstance(context_data, str) else context_data
                    user_location = context_dict.get('user_profile', {}).get('location')
                    if user_location:
                        recommendations.append(f"For your location ({user_location}):")
                        recommendations.append("• Consult local agricultural extension services")
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            synthesis = {
                "recommendations": "\n".join(recommendations) if recommendations else "No specific recommendations available.",
                "sources_consulted": sources,
                "confidence": 0.8 if recommendations else 0.3,
                "next_steps": [
                    "Implement recommended practices gradually",
                    "Monitor crop response to interventions", 
                    "Keep detailed records for future reference"
                ]
            }
            
            return json.dumps(synthesis)
            
        except Exception as e:
            logger.error(f"❌ Recommendation synthesis failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "recommendations": "Unable to synthesize recommendations",
                "sources_consulted": [],
                "confidence": 0.0
            })
    
    async def _preprocess_and_cache_datasets(self, priority_datasets: List[str] = None) -> Dict[str, Any]:
        """
        Preprocess and cache priority agricultural datasets from Hugging Face
        
        Args:
            priority_datasets: List of dataset names to prioritize for caching
            
        Returns:
            Dictionary with caching results and statistics
        """
        try:
            if not priority_datasets:
                priority_datasets = [
                    "Mahesh2841/Agriculture",
                    "Dharine/agriculture-10k"
                ]
            
            caching_results = {
                "cached_datasets": [],
                "failed_datasets": [],
                "total_documents": 0,
                "processing_time": 0
            }
            
            start_time = datetime.utcnow()
            
            for dataset_name in priority_datasets:  # Cache all specified datasets
                try:
                    logger.info(f"📦 Caching dataset: {dataset_name}")
                    
                    # Load dataset with streaming for large datasets
                    if self.dependencies_available and 'load_dataset' in self.dependencies:
                        load_dataset = self.dependencies['load_dataset']
                        dataset = load_dataset(dataset_name, split="train", streaming=True)
                    else:
                        raise ImportError("load_dataset not available")
                    
                    # Cache the dataset
                    self.hf_datasets_cache[dataset_name] = dataset
                    
                    # Process and embed a sample for quick access
                    sample_docs = []
                    for i, item in enumerate(dataset):
                        if i >= 100:  # Cache first 100 items for quick access
                            break
                        
                        # Extract meaningful content
                        processed_item = self._extract_content_from_dataset_item(item, dataset_name)
                        if processed_item:
                            sample_docs.append(processed_item)
                    
                    # Store processed sample
                    cache_key = f"{dataset_name}_processed_sample"
                    self.hf_datasets_cache[cache_key] = sample_docs
                    
                    # CRITICAL FIX: Actually populate ChromaDB with the agricultural data
                    if self.chroma_client and sample_docs:
                        await self._populate_chromadb_with_docs(sample_docs, dataset_name)
                    
                    caching_results["cached_datasets"].append({
                        "name": dataset_name,
                        "sample_size": len(sample_docs),
                        "status": "success"
                    })
                    
                    caching_results["total_documents"] += len(sample_docs)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cache dataset {dataset_name}: {str(e)}")
                    caching_results["failed_datasets"].append({
                        "name": dataset_name,
                        "error": str(e)
                    })
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            caching_results["processing_time"] = processing_time
            
            logger.info(f"✅ Dataset caching completed: {len(caching_results['cached_datasets'])} datasets cached")
            return caching_results
            
        except Exception as e:
            logger.error(f"❌ Dataset preprocessing failed: {str(e)}")
            return {
                "cached_datasets": [],
                "failed_datasets": [],
                "total_documents": 0,
                "error": str(e)
            }
    
    async def populate_chromadb_with_hf_datasets(self, max_documents_per_dataset: int = 1000) -> Dict[str, Any]:
        """
        Populate ChromaDB with content from Hugging Face datasets
        
        Args:
            max_documents_per_dataset: Maximum number of documents to process per dataset
            
        Returns:
            Dictionary with population results and statistics
        """
        try:
            if not self.chroma_client:
                return {
                    "error": "ChromaDB client not available",
                    "documents_indexed": 0,
                    "datasets_processed": 0
                }
                
            logger.info("🚀 Starting ChromaDB population with Hugging Face datasets...")
            
            # Initialize embeddings if not available
            if not self.langchain_embeddings:
                logger.info("🔧 Initializing OpenAI embeddings for population...")
                from langchain_community.embeddings import OpenAIEmbeddings
                self.langchain_embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model=self.embedding_model
                )
            
            population_results = {
                "documents_indexed": 0,
                "datasets_processed": 0,
                "processing_time": 0,
                "dataset_results": []
            }
            
            start_time = datetime.utcnow()
            
            # Process each dataset
            datasets_to_process = [
                "Mahesh2841/Agriculture",
                "Dharine/agriculture-10k"
            ]
            
            for dataset_name in datasets_to_process:
                try:
                    logger.info(f"📦 Processing dataset: {dataset_name}")
                    
                    # Load dataset
                    if self.dependencies_available and 'load_dataset' in self.dependencies:
                        load_dataset = self.dependencies['load_dataset']
                        dataset = load_dataset(dataset_name, split="train", streaming=True)
                    else:
                        raise ImportError("load_dataset not available")
                    
                    documents_to_index = []
                    processed_count = 0
                    
                    # Process dataset items
                    for item in dataset:
                        if processed_count >= max_documents_per_dataset:
                            break
                        
                        # Extract content from dataset item
                        processed_item = self._extract_content_from_dataset_item(item, dataset_name)
                        if processed_item and processed_item.get("content"):
                            # Create LangChain document
                            from langchain.schema import Document
                            
                            doc = Document(
                                page_content=processed_item["content"],
                                metadata={
                                    "source_dataset": dataset_name,
                                    "title": processed_item.get("title", ""),
                                    "crop_type": processed_item.get("crop_type", "general"),
                                    "region": processed_item.get("region", "global"),
                                    "document_type": "huggingface_dataset",
                                    "processed_at": datetime.utcnow().isoformat(),
                                    "relevance_score": processed_item.get("relevance_score", 0.5)
                                }
                            )
                            documents_to_index.append(doc)
                            processed_count += 1
                    
                    # Add documents to ChromaDB
                    if documents_to_index:
                        logger.info(f"📝 Indexing {len(documents_to_index)} documents from {dataset_name}")
                        if self.langchain_vectorstore is None:
                            logger.error("❌ LangChain vectorstore not initialized")
                            raise Exception("LangChain vectorstore not initialized")
                        self.langchain_vectorstore.add_documents(documents_to_index)
                        
                        population_results["dataset_results"].append({
                            "dataset_name": dataset_name,
                            "documents_indexed": len(documents_to_index),
                            "status": "success"
                        })
                        
                        population_results["documents_indexed"] += len(documents_to_index)
                        population_results["datasets_processed"] += 1
                        
                        logger.info(f"✅ Successfully indexed {len(documents_to_index)} documents from {dataset_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process dataset {dataset_name}: {str(e)}")
                    population_results["dataset_results"].append({
                        "dataset_name": dataset_name,
                        "documents_indexed": 0,
                        "status": "failed",
                        "error": str(e)
                    })
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            population_results["processing_time"] = processing_time
            
            logger.info(f"🎉 ChromaDB population completed: {population_results['documents_indexed']} documents indexed from {population_results['datasets_processed']} datasets")
            return population_results
            
        except Exception as e:
            logger.error(f"❌ ChromaDB population failed: {str(e)}")
            return {
                "error": str(e),
                "documents_indexed": 0,
                "datasets_processed": 0
            }
    
    def _extract_content_from_dataset_item(self, item: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract meaningful content from a dataset item with standardized 2-column structure
        
        Args:
            item: Individual item from the dataset
            dataset_name: Name of the source dataset
            
        Returns:
            Processed item with standardized content structure
        """
        try:
            content = ""
            title = ""
            metadata = {"source_dataset": dataset_name}
            
            # Standardize to 2 columns by removing first column if needed
            item_keys = list(item.keys())
            
            # Handle Mahesh2841/Agriculture dataset
            if "mahesh2841" in dataset_name.lower() or "agriculture" in dataset_name.lower():
                # Standardize to 2 columns - use last 2 columns
                if len(item_keys) > 2:
                    # Remove first column, keep last 2
                    standardized_item = {item_keys[-2]: item[item_keys[-2]], item_keys[-1]: item[item_keys[-1]]}
                else:
                    standardized_item = item
                
                # Extract content from standardized structure
                values = list(standardized_item.values())
                if len(values) >= 2:
                    content = str(values[1])  # Second column as content
                    title = str(values[0])    # First column as title
                    
                    # Add metadata from original item if available
                    metadata.update({
                        "original_columns": len(item_keys),
                        "standardized": len(item_keys) > 2,
                        "dataset_type": "agriculture_mahesh"
                    })
            
            # Handle Dharine/agriculture-10k dataset
            elif "dharine" in dataset_name.lower() or "agriculture-10k" in dataset_name.lower():
                # Standardize to 2 columns - use last 2 columns
                if len(item_keys) > 2:
                    # Remove first column, keep last 2
                    standardized_item = {item_keys[-2]: item[item_keys[-2]], item_keys[-1]: item[item_keys[-1]]}
                else:
                    standardized_item = item
                
                # Extract content from standardized structure
                values = list(standardized_item.values())
                if len(values) >= 2:
                    content = str(values[1])  # Second column as content
                    title = str(values[0])    # First column as title
                    
                    # Add metadata from original item if available
                    metadata.update({
                        "original_columns": len(item_keys),
                        "standardized": len(item_keys) > 2,
                        "dataset_type": "agriculture_10k"
                    })
            
            else:
                # Fallback for other datasets - try to maintain 2-column structure
                if len(item_keys) > 2:
                    # Use last 2 columns
                    standardized_item = {item_keys[-2]: item[item_keys[-2]], item_keys[-1]: item[item_keys[-1]]}
                    values = list(standardized_item.values())
                    if len(values) >= 2:
                        content = str(values[1])
                        title = str(values[0])
                else:
                    # Use available columns
                    values = list(item.values())
                    if len(values) >= 2:
                        content = str(values[1])
                        title = str(values[0])
                    elif len(values) == 1:
                        content = str(values[0])
                        title = "Agricultural Data"
                
                metadata.update({
                    "original_columns": len(item_keys),
                    "standardized": len(item_keys) > 2,
                    "dataset_type": "generic"
                })
            
            # Validate content
            if not content or len(content.strip()) < 5:
                return None
            
            # Clean and format content
            content = content.strip()
            title = title.strip() if title else "Agricultural Information"
            
            # Limit content length for performance
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            return {
                "content": content,
                "title": title,
                "metadata": metadata,
                "source_dataset": dataset_name,
                "column_count": len(item_keys),
                "standardized": len(item_keys) > 2
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract content from item: {str(e)}")
            return None

    async def _tool_fetch_huggingface_data(
        self,
        query: str,
        dataset_name: Optional[str] = None,
        crop_type: Optional[str] = None,
        limit: int = 5
    ) -> str:
        """
        Tool function for fetching agricultural data from Hugging Face datasets
        
        Args:
            query: Search query for agricultural information
            dataset_name: Optional specific dataset to search
            crop_type: Optional crop type for filtering
            limit: Maximum results to return
            
        Returns:
            JSON string with Hugging Face data results
        """
        try:
            if not self.hf_api:
                return json.dumps({
                    "error": "Hugging Face API not available",
                    "documents": [],
                    "sources": [],
                    "dataset_info": {}
                })
            
            results = []
            sources = []
            
            # Define agricultural datasets to search
            target_datasets = []
            if dataset_name:
                target_datasets = [dataset_name]
            else:
                # User-specified agricultural datasets (standardized to 2 columns)
                target_datasets = [
                    "Mahesh2841/Agriculture",  # Primary agriculture dataset
                    "Dharine/agriculture-10k"  # Agriculture 10k dataset
                ]
            
            for dataset_id in target_datasets:  # Process all specified datasets
                try:
                    # First, try to use cached processed sample for faster search
                    cache_key = f"{dataset_id}_processed_sample"
                    if cache_key in self.hf_datasets_cache:
                        logger.info(f"🔍 Using cached sample for {dataset_id}")
                        cached_sample = self.hf_datasets_cache[cache_key]
                        
                        # Search within cached processed sample
                        search_results = self._search_processed_sample(cached_sample, query, crop_type, limit)
                        
                        for result in search_results:
                            results.append({
                                "content": result.get("content", ""),
                                "title": result.get("title", ""),
                                "source_dataset": dataset_id,
                                "relevance_score": result.get("score", 0.5),
                                "metadata": result.get("metadata", {}),
                                "cached_search": True
                            })
                            sources.append(f"HF:{dataset_id}")
                            
                            if len(results) >= limit:
                                break
                    
                    else:
                        # Fallback to live dataset search
                        logger.info(f"📡 Live searching dataset {dataset_id}")
                        
                        # Load dataset
                        if dataset_id in self.hf_datasets_cache:
                            dataset = self.hf_datasets_cache[dataset_id]
                        else:
                            # Load a small subset to avoid memory issues
                            if self.dependencies_available and 'load_dataset' in self.dependencies:
                                load_dataset = self.dependencies['load_dataset']
                                dataset = load_dataset(dataset_id, split="train", streaming=True)
                                self.hf_datasets_cache[dataset_id] = dataset
                            else:
                                raise ImportError("load_dataset not available")
                        
                        # Search within dataset using original method
                        search_results = self._search_hf_dataset(dataset, query, crop_type, limit)
                        
                        for result in search_results:
                            results.append({
                                "content": result.get("content", ""),
                                "title": result.get("title", ""),
                                "source_dataset": dataset_id,
                                "relevance_score": result.get("score", 0.5),
                                "cached_search": False
                            })
                            sources.append(f"HF:{dataset_id}")
                            
                            if len(results) >= limit:
                                break
                    
                    if len(results) >= limit:
                        break
                        
                except Exception as dataset_error:
                    logger.warning(f"⚠️ Failed to search dataset {dataset_id}: {str(dataset_error)}")
                    continue
            
            return json.dumps({
                "documents": results[:limit],
                "sources": sources[:limit],
                "dataset_info": {
                    "searched_datasets": target_datasets,
                    "total_results": len(results)
                },
                "query_used": query
            })
            
        except Exception as e:
            logger.error(f"❌ Hugging Face data fetch failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "documents": [],
                "sources": [],
                "dataset_info": {}
            })
    
    async def _tool_populate_knowledge_base(
        self,
        max_documents_per_dataset: int = 1000,
        force_rebuild: bool = False
    ) -> str:
        """
        Tool function for populating ChromaDB with Hugging Face agricultural datasets
        
        Args:
            max_documents_per_dataset: Maximum number of documents to process per dataset
            force_rebuild: Whether to rebuild the knowledge base from scratch
            
        Returns:
            JSON string with population results
        """
        try:
            logger.info(f"🚀 Starting knowledge base population with {max_documents_per_dataset} docs per dataset")
            
            # Call the main population method
            results = await self.populate_chromadb_with_hf_datasets(max_documents_per_dataset)
            
            return json.dumps({
                "success": True,
                "documents_indexed": results.get("documents_indexed", 0),
                "datasets_processed": results.get("datasets_processed", 0),
                "processing_time_ms": results.get("processing_time", 0),
                "dataset_results": results.get("dataset_results", []),
                "message": f"Successfully indexed {results.get('documents_indexed', 0)} documents from {results.get('datasets_processed', 0)} datasets"
            })
            
        except Exception as e:
            logger.error(f"❌ Knowledge base population tool failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "documents_indexed": 0,
                "datasets_processed": 0,
                "message": f"Failed to populate knowledge base: {str(e)}"
            })
    
    def _search_processed_sample(self, processed_sample: List[Dict[str, Any]], query: str, crop_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search within preprocessed and cached sample data for faster results
        
        Args:
            processed_sample: List of preprocessed dataset items
            query: Search query
            crop_type: Optional crop type filter
            limit: Maximum results
            
        Returns:
            List of relevant documents with enhanced scoring
        """
        results = []
        query_lower = query.lower()
        crop_filter = crop_type.lower() if crop_type else ""
        
        try:
            for item in processed_sample:
                content = item.get("content", "").lower()
                title = item.get("title", "").lower()
                metadata = item.get("metadata", {})
                
                # Enhanced relevance scoring
                score = 0.0
                
                # Query term matching with weights
                query_terms = query_lower.split()
                for term in query_terms:
                    # Title matching (highest weight)
                    if term in title:
                        score += 1.0
                    
                    # Content matching (medium weight)
                    content_matches = content.count(term)
                    score += content_matches * 0.3
                    
                    # Metadata matching (medium weight)
                    for meta_value in metadata.values():
                        if isinstance(meta_value, str) and term in str(meta_value).lower():
                            score += 0.5
                
                # Agricultural relevance boost
                agricultural_terms = [
                    "crop", "plant", "farm", "soil", "pest", "disease", "yield", "harvest",
                    "irrigation", "fertilizer", "seed", "growth", "cultivation", "agriculture",
                    "farming", "nutrition", "weather", "climate", "season"
                ]
                
                for term in agricultural_terms:
                    if term in content or term in title:
                        score += 0.3
                
                # Crop-specific boost
                if crop_filter:
                    crop_variants = [crop_filter, crop_filter + "s"]
                    if crop_filter.endswith('s'):
                        crop_variants.append(crop_filter[:-1])
                    for variant in crop_variants:
                        if variant in content or variant in title:
                            score += 1.5
                        
                        # Check metadata for crop information
                        metadata_crop = metadata.get("crop", "").lower()
                        if variant in metadata_crop:
                            score += 2.0
                
                # Dataset-specific scoring adjustments
                source_dataset = metadata.get("source_dataset", "")
                if "agriculture" in source_dataset or "crop" in source_dataset or "fao" in source_dataset:
                    score += 0.5
                
                # Quality filters
                if len(item.get("content", "")) < 20:  # Too short
                    score *= 0.5
                
                if score > 0.5:  # Minimum relevance threshold
                    results.append({
                        "content": item.get("content", ""),
                        "title": item.get("title", ""),
                        "score": min(score, 5.0),  # Cap score at 5.0
                        "metadata": metadata
                    })
            
            # Sort by relevance score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Processed sample search failed: {str(e)}")
            return []

    def _search_hf_dataset(self, dataset, query: str, crop_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a Hugging Face dataset for relevant content
        
        Args:
            dataset: Loaded HF dataset
            query: Search query
            crop_type: Optional crop type filter
            limit: Maximum results
            
        Returns:
            List of relevant documents
        """
        results = []
        query_lower = query.lower()
        crop_filter = crop_type.lower() if crop_type else ""
        
        try:
            # Iterate through dataset (streaming)
            for i, item in enumerate(dataset):
                if i >= 1000:  # Limit search to first 1000 items to avoid timeout
                    break
                
                # Extract text content based on dataset structure
                content = ""
                title = ""
                
                if "abstract" in item:
                    content = item["abstract"]
                    title = item.get("title", "")
                elif "text" in item:
                    content = item["text"]
                    title = item.get("title", content[:100] + "...")
                elif "passage" in item:
                    content = item["passage"]
                    title = item.get("query", content[:100] + "...")
                else:
                    # Try to find any text field
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 50:
                            content = value
                            title = key.replace("_", " ").title()
                            break
                
                if not content:
                    continue
                
                content_lower = content.lower()
                title_lower = title.lower()
                
                # Simple relevance scoring
                score = 0.0
                
                # Check for query terms
                query_terms = query_lower.split()
                for term in query_terms:
                    if term in content_lower:
                        score += 0.3
                    if term in title_lower:
                        score += 0.5
                
                # Check for agricultural terms
                agricultural_terms = ["crop", "plant", "farm", "soil", "pest", "disease", "yield", "harvest"]
                for term in agricultural_terms:
                    if term in content_lower:
                        score += 0.2
                
                # Check for crop-specific terms
                if crop_filter and len(crop_filter) > 0:
                    if crop_filter in content_lower or crop_filter in title_lower:
                        score += 0.4
                
                # Only include if above threshold
                if score > 0.3:
                    results.append({
                        "content": content[:500] + "..." if len(content) > 500 else content,
                        "title": title,
                        "score": min(score, 1.0)
                    })
                
                if len(results) >= limit:
                    break
            
            # Sort by relevance score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Dataset search failed: {str(e)}")
            return []
    
    async def _geocode_location_with_opencage(self, location: str) -> Optional[tuple]:
        """
        Geocode a location using OpenCage Geocoding API
        
        Args:
            location: Location name to geocode
            
        Returns:
            Tuple of (latitude, longitude) or None if geocoding fails
        """
        try:
            # Check if OpenCage API key is available
            if not settings.OPENCAGE_API_KEY:
                logger.warning("⚠️ OpenCage API key not configured")
                return None
                
            from opencage.geocoder import OpenCageGeocode
            from opencage.geocoder import InvalidInputError, RateLimitExceededError, UnknownError
            
            # Initialize OpenCage Geocoder
            key = settings.OPENCAGE_API_KEY
            geocoder = OpenCageGeocode(key)
            
            # Use OpenCage to geocode
            results = geocoder.geocode(location)
            
            if results and len(results) > 0:
                latitude = results[0]['geometry']['lat']
                longitude = results[0]['geometry']['lng']
                return (latitude, longitude)
            
            return None
            
        except InvalidInputError as e:
            logger.error(f"❌ OpenCage invalid input error: {str(e)}")
            return None
        except RateLimitExceededError as e:
            logger.error(f"❌ OpenCage rate limit exceeded: {str(e)}")
            return None
        except UnknownError as e:
            logger.error(f"❌ OpenCage unknown error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Geocoding failed: {str(e)}")
            return None
    
    async def _generate_regional_recommendations(
        self, 
        lat: float, 
        lon: float, 
        location: str, 
        crop_type: str = None
    ) -> List[str]:
        """
        Generate region-specific agricultural recommendations based on coordinates
        
        Args:
            lat: Latitude
            lon: Longitude  
            location: Original location name
            crop_type: Optional crop type
            
        Returns:
            List of regional recommendations
        """
        try:
            recommendations = []
            
            # Determine general climate zone based on latitude
            if abs(lat) < 23.5:  # Tropical zone
                recommendations.extend([
                    "High humidity and consistent temperatures year-round",
                    "Monitor for tropical diseases and pests",
                    "Consider multiple growing seasons per year",
                    "Ensure adequate drainage during rainy seasons"
                ])
            elif abs(lat) < 35:  # Subtropical zone
                recommendations.extend([
                    "Moderate seasonal variation with warm summers",
                    "Plan for potential frost in winter months",
                    "Take advantage of extended growing seasons",
                    "Monitor for both temperate and tropical pest pressures"
                ])
            elif abs(lat) < 50:  # Temperate zone
                recommendations.extend([
                    "Distinct seasonal changes with cold winters",
                    "Plan planting around last frost dates",
                    "Consider cold-hardy crop varieties",
                    "Prepare for seasonal pest and disease cycles"
                ])
            else:  # Cold zone
                recommendations.extend([
                    "Short growing season with potential for frost",
                    "Focus on cold-tolerant and fast-maturing varieties",
                    "Consider protected growing environments",
                    "Plan for winter storage and preservation"
                ])
            
            # Add crop-specific recommendations if provided
            if crop_type:
                crop_recommendations = self._get_crop_specific_regional_advice(crop_type, lat)
                recommendations.extend(crop_recommendations)
            
            # Add general location-based advice
            recommendations.extend([
                f"Consult local agricultural extension services in {location}",
                "Monitor local weather patterns and seasonal variations",
                "Connect with nearby farmers for regional best practices",
                "Consider soil testing for location-specific conditions"
            ])
            
            return recommendations[:8]  # Limit to 8 recommendations
            
        except Exception as e:
            logger.error(f"❌ Regional recommendations generation failed: {str(e)}")
            return [
                "Consult local agricultural extension services",
                "Monitor local weather conditions", 
                "Follow regional farming best practices"
            ]
    
    def _get_crop_specific_regional_advice(self, crop_type: str, lat: float) -> List[str]:
        """
        Get crop-specific advice based on latitude/climate zone
        
        Args:
            crop_type: Type of crop
            lat: Latitude for climate zone determination
            
        Returns:
            List of crop-specific regional recommendations
        """
        crop_advice = {
            "rice": {
                "tropical": ["Ideal conditions for year-round rice production", "Monitor for rice blast disease"],
                "temperate": ["Single season rice production", "Ensure adequate water management"],
                "cold": ["Consider short-season rice varieties", "May require greenhouse cultivation"]
            },
            "corn": {
                "tropical": ["Multiple corn harvests possible", "Watch for corn borer infestations"], 
                "temperate": ["Plant after soil reaches 60°F", "Monitor for corn smut and rootworm"],
                "cold": ["Choose early-maturing varieties", "Protect from late spring frosts"]
            },
            "wheat": {
                "tropical": ["Consider spring wheat varieties", "Monitor for rust diseases"],
                "temperate": ["Winter wheat performs well", "Plan for fall planting"],
                "cold": ["Spring wheat recommended", "Short growing season varieties"]
            }
        }
        
        # Determine climate zone
        if abs(lat) < 35:
            zone = "tropical"
        elif abs(lat) < 50:
            zone = "temperate" 
        else:
            zone = "cold"
        
        return crop_advice.get(crop_type.lower(), {}).get(zone, [f"Adapt {crop_type} cultivation to local conditions"])
    
    async def _get_weather_data(self, lat: float, lon: float, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Get weather data using OpenWeatherMap API
        
        Args:
            lat: Latitude
            lon: Longitude
            days_ahead: Number of days to forecast
            
        Returns:
            Dictionary with current and forecast weather data
        """
        try:
            import aiohttp
            
            if not settings.WEATHER_API_KEY:
                logger.warning("⚠️ Weather API key not configured, using mock data")
                return self._get_mock_weather_data(lat, lon)
            
            base_url = settings.WEATHER_API_URL
            api_key = settings.WEATHER_API_KEY
            
            async with aiohttp.ClientSession() as session:
                # Get current weather
                current_url = f"{base_url}/weather"
                current_params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": api_key,
                    "units": "metric"
                }
                
                async with session.get(current_url, params=current_params) as response:
                    if response.status == 200:
                        current_data = await response.json()
                    else:
                        logger.error(f"❌ Current weather API error: {response.status}")
                        return self._get_mock_weather_data(lat, lon)
                
                # Get forecast data
                forecast_url = f"{base_url}/forecast"
                forecast_params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": api_key,
                    "units": "metric",
                    "cnt": days_ahead * 8  # 3-hour intervals, 8 per day
                }
                
                async with session.get(forecast_url, params=forecast_params) as response:
                    if response.status == 200:
                        forecast_data = await response.json()
                    else:
                        logger.error(f"❌ Forecast weather API error: {response.status}")
                        forecast_data = {"list": []}
                
                # Process and structure the weather data
                structured_data = {
                    "current": {
                        "temperature": current_data["main"]["temp"],
                        "humidity": current_data["main"]["humidity"],
                        "pressure": current_data["main"]["pressure"],
                        "description": current_data["weather"][0]["description"],
                        "wind_speed": current_data.get("wind", {}).get("speed", 0),
                        "precipitation": current_data.get("rain", {}).get("1h", 0) + current_data.get("snow", {}).get("1h", 0),
                        "feels_like": current_data["main"]["feels_like"],
                        "visibility": current_data.get("visibility", 10000) / 1000  # Convert to km
                    },
                    "forecast": [],
                    "daily_summary": []
                }
                
                # Process forecast data into daily summaries
                daily_data = {}
                for item in forecast_data.get("list", []):
                    date = item["dt_txt"].split(" ")[0]  # Extract date
                    
                    if date not in daily_data:
                        daily_data[date] = {
                            "temps": [],
                            "humidity": [],
                            "precipitation": 0,
                            "descriptions": []
                        }
                    
                    daily_data[date]["temps"].append(item["main"]["temp"])
                    daily_data[date]["humidity"].append(item["main"]["humidity"])
                    daily_data[date]["precipitation"] += item.get("rain", {}).get("3h", 0) + item.get("snow", {}).get("3h", 0)
                    daily_data[date]["descriptions"].append(item["weather"][0]["description"])
                
                # Create daily summaries
                for date, data in list(daily_data.items())[:days_ahead]:
                    structured_data["daily_summary"].append({
                        "date": date,
                        "temp_high": max(data["temps"]),
                        "temp_low": min(data["temps"]),
                        "avg_humidity": sum(data["humidity"]) / len(data["humidity"]),
                        "total_precipitation": data["precipitation"],
                        "description": max(set(data["descriptions"]), key=data["descriptions"].count)
                    })
                
                return structured_data
                
        except Exception as e:
            logger.error(f"❌ Weather API failed: {str(e)}")
            return self._get_mock_weather_data(lat, lon)
    
    def _get_mock_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Generate mock weather data when API is unavailable
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Mock weather data structure
        """
        import random
        
        # Generate realistic data based on latitude (climate zone)
        if abs(lat) < 23.5:  # Tropical
            base_temp = random.uniform(25, 35)
            humidity = random.uniform(70, 90)
        elif abs(lat) < 35:  # Subtropical
            base_temp = random.uniform(20, 30)
            humidity = random.uniform(50, 80)
        elif abs(lat) < 50:  # Temperate
            base_temp = random.uniform(15, 25)
            humidity = random.uniform(40, 70)
        else:  # Cold
            base_temp = random.uniform(5, 15)
            humidity = random.uniform(30, 60)
        
        return {
            "current": {
                "temperature": base_temp,
                "humidity": humidity,
                "pressure": random.uniform(1000, 1020),
                "description": "partly cloudy",
                "wind_speed": random.uniform(5, 15),
                "precipitation": random.uniform(0, 5),
                "feels_like": base_temp + random.uniform(-2, 2),
                "visibility": random.uniform(8, 15)
            },
            "forecast": [],
            "daily_summary": [
                {
                    "date": f"2024-01-{i+1:02d}",
                    "temp_high": base_temp + random.uniform(2, 8),
                    "temp_low": base_temp - random.uniform(2, 8),
                    "avg_humidity": humidity + random.uniform(-10, 10),
                    "total_precipitation": random.uniform(0, 10),
                    "description": random.choice(["sunny", "partly cloudy", "cloudy", "light rain"])
                }
                for i in range(7)
            ],
            "_mock_data": True
        }
    
    async def _generate_weather_recommendations(
        self, 
        weather_data: Dict[str, Any], 
        crop_type: str = None
    ) -> List[str]:
        """
        Generate weather-based agricultural recommendations
        
        Args:
            weather_data: Current and forecast weather data
            crop_type: Optional crop type for specific recommendations
            
        Returns:
            List of weather-based recommendations
        """
        try:
            recommendations = []
            current = weather_data.get("current", {})
            daily_summary = weather_data.get("daily_summary", [])
            
            # Current weather recommendations
            temp = current.get("temperature", 20)
            humidity = current.get("humidity", 50)
            precipitation = current.get("precipitation", 0)
            wind_speed = current.get("wind_speed", 5)
            
            # Temperature-based recommendations
            if temp < 5:
                recommendations.append("🥶 Very cold conditions - protect crops from frost damage")
                recommendations.append("Consider covering sensitive plants or moving to protected areas")
            elif temp < 10:
                recommendations.append("❄️ Cold weather - monitor for potential frost")
                recommendations.append("Delay planting of warm-season crops")
            elif temp > 35:
                recommendations.append("🌡️ Very hot conditions - ensure adequate irrigation")
                recommendations.append("Provide shade for sensitive crops if possible")
            elif temp > 30:
                recommendations.append("☀️ Hot weather - increase watering frequency")
                recommendations.append("Monitor crops for heat stress signs")
            else:
                recommendations.append("🌤️ Favorable temperature conditions for most agricultural activities")
            
            # Humidity-based recommendations
            if humidity > 80:
                recommendations.append("💧 High humidity - monitor for fungal diseases")
                recommendations.append("Ensure good air circulation around plants")
            elif humidity < 30:
                recommendations.append("🏜️ Low humidity - increase irrigation and consider mulching")
            
            # Precipitation-based recommendations
            if precipitation > 10:
                recommendations.append("🌧️ Heavy precipitation - ensure proper drainage")
                recommendations.append("Delay field work until soil conditions improve")
            elif precipitation > 2:
                recommendations.append("🌦️ Light to moderate precipitation - good for plant growth")
                recommendations.append("Monitor soil moisture levels")
            else:
                recommendations.append("☀️ Dry conditions - check irrigation needs")
            
            # Wind-based recommendations
            if wind_speed > 20:
                recommendations.append("💨 Strong winds - protect tall crops and check for damage")
                recommendations.append("Delay spraying operations due to wind")
            
            # Forecast-based recommendations
            if daily_summary:
                upcoming_rain = sum(day.get("total_precipitation", 0) for day in daily_summary[:3])
                if upcoming_rain > 20:
                    recommendations.append("📅 Heavy rain expected - plan drainage and harvest timing")
                elif upcoming_rain < 2:
                    recommendations.append("📅 Dry period ahead - plan irrigation schedule")
                
                temp_range = [day.get("temp_high", 20) for day in daily_summary[:3]]
                if max(temp_range) > 35:
                    recommendations.append("📅 Heat wave expected - prepare cooling measures")
                elif min(temp_range) < 5:
                    recommendations.append("📅 Cold snap expected - prepare frost protection")
            
            # Crop-specific recommendations
            if crop_type:
                crop_recommendations = self._get_weather_crop_specific_advice(
                    crop_type, current, daily_summary
                )
                recommendations.extend(crop_recommendations)
            
            return recommendations[:8]  # Limit to 8 recommendations
            
        except Exception as e:
            logger.error(f"❌ Weather recommendations generation failed: {str(e)}")
            return [
                "Monitor local weather conditions regularly",
                "Adjust farming activities based on weather patterns",
                "Consult local weather forecasts for planning"
            ]
    
    def _get_weather_crop_specific_advice(
        self, 
        crop_type: str, 
        current_weather: Dict[str, Any], 
        forecast: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Get crop-specific weather advice
        
        Args:
            crop_type: Type of crop
            current_weather: Current weather conditions
            forecast: Weather forecast data
            
        Returns:
            List of crop-specific weather recommendations
        """
        recommendations = []
        temp = current_weather.get("temperature", 20)
        humidity = current_weather.get("humidity", 50)
        
        crop_weather_advice = {
            "rice": {
                "temp_advice": {
                    (temp < 15): "Rice growth severely limited by cold temperatures",
                    (temp > 35): "High temperatures may reduce rice yield - ensure adequate water",
                    (20 <= temp <= 30): "Optimal temperature range for rice cultivation"
                },
                "humidity_advice": {
                    (humidity > 80): "High humidity ideal for rice but monitor for blast disease",
                    (humidity < 60): "Rice prefers higher humidity - maintain water levels"
                }
            },
            "corn": {
                "temp_advice": {
                    (temp < 10): "Too cold for corn planting - wait for warmer weather",
                    (temp > 32): "Hot weather stress - ensure adequate soil moisture for corn",
                    (15 <= temp <= 25): "Good temperature range for corn growth"
                },
                "humidity_advice": {
                    (humidity > 85): "High humidity may promote corn fungal diseases",
                    (humidity < 40): "Corn may experience drought stress - increase irrigation"
                }
            },
            "wheat": {
                "temp_advice": {
                    (temp < 5): "Wheat can tolerate cold but monitor for extreme frost",
                    (temp > 30): "Heat stress during grain filling - monitor yield impact",
                    (10 <= temp <= 25): "Favorable temperatures for wheat development"
                },
                "humidity_advice": {
                    (humidity > 80): "High humidity increases wheat rust and mildew risk",
                    (humidity < 50): "Wheat handles moderate humidity well"
                }
            }
        }
        
        crop_advice = crop_weather_advice.get(crop_type.lower(), {})
        
        # Temperature advice
        for condition, advice in crop_advice.get("temp_advice", {}).items():
            if condition:
                recommendations.append(f"🌾 {advice}")
                break
        
        # Humidity advice
        for condition, advice in crop_advice.get("humidity_advice", {}).items():
            if condition:
                recommendations.append(f"💧 {advice}")
                break
        
        return recommendations
    
    async def _tool_langchain_enhanced_search(
        self,
        query: str,
        search_type: str = "similarity",
        crop_type: Optional[str] = None,
        include_sources: bool = True
    ) -> str:
        """
        Enhanced knowledge search using LangChain with advanced retrieval capabilities
        
        Args:
            query: Search query
            search_type: Type of search (similarity, mmr, similarity_score_threshold)
            crop_type: Optional crop type filter
            include_sources: Whether to include source documents
            
        Returns:
            JSON string with enhanced search results
        """
        try:
            if not self.langchain_vectorstore or not self.retrieval_qa_chain:
                return json.dumps({
                    "error": "LangChain components not available",
                    "documents": [],
                    "sources": []
                })
            
            # Build search query with crop type filter
            search_query = query
            if crop_type:
                search_query = f"{query} related to {crop_type} cultivation"
            
            # Configure retriever based on search type
            search_kwargs = {"k": self.max_search_results}
            
            # Only add score_threshold for compatible search types
            if search_type == "similarity_score_threshold":
                search_kwargs["filter_threshold"] = self.similarity_threshold
                
            retriever = self.langchain_vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            # Perform enhanced search using LangChain
            if search_type == "mmr":
                # Maximum Marginal Relevance search for diverse results
                docs = retriever.get_relevant_documents(search_query)
            else:
                # Standard similarity search
                docs = retriever.get_relevant_documents(search_query)
            
            # Process results
            processed_docs = []
            sources = []
            
            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata
                
                processed_docs.append({
                    "content": content,
                    "source": metadata.get("source", "Unknown"),
                    "crop_type": metadata.get("crop_type", "general"),
                    "region": metadata.get("region", "global"),
                    "document_type": metadata.get("document_type", "unknown"),
                    "relevance_score": getattr(doc, 'score', 0.8)
                })
                
                sources.append(metadata.get("source", "Unknown"))
            
            # Use retrieval QA chain for enhanced answer generation
            if self.retrieval_qa_chain and callable(self.retrieval_qa_chain):
                try:
                    qa_result = self.retrieval_qa_chain({"query": search_query})
                except Exception as e:
                    logger.warning(f"⚠️ Retrieval QA chain failed: {e}")
                    qa_result = {"result": "Enhanced answer generation temporarily unavailable", "source_documents": []}
            else:
                qa_result = {"result": "Enhanced answer generation not available", "source_documents": []}
            
            result = {
                "documents": processed_docs,
                "sources": sources,
                "enhanced_answer": qa_result.get("result", ""),
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc in qa_result.get("source_documents", [])
                ] if include_sources else [],
                "search_type": search_type,
                "total_results": len(processed_docs),
                "langchain_enhanced": True
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"❌ LangChain enhanced search failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "documents": [],
                "sources": [],
                "langchain_enhanced": False
            })
    
    async def _tool_manage_hf_datasets(
        self,
        action: str,
        dataset_names: Optional[str] = None,
        priority_level: str = "medium"
    ) -> str:
        """
        Tool function for managing Hugging Face datasets
        
        Args:
            action: Action to perform (cache_datasets, get_stats, refresh_cache, list_available)
            dataset_names: Optional specific dataset names
            priority_level: Priority level for operations
            
        Returns:
            JSON string with management results
        """
        try:
            if action == "cache_datasets":
                if dataset_names:
                    # Parse comma-separated dataset names
                    dataset_list = [name.strip() for name in dataset_names.split(',') if name.strip()]
                    result = await self._preprocess_and_cache_datasets(dataset_list)
                else:
                    # Use priority-based default datasets
                    priority_datasets = {
                        "high": ["Mahesh2841/Agriculture", "Dharine/agriculture-10k"],
                        "medium": ["Mahesh2841/Agriculture"],
                        "low": ["Dharine/agriculture-10k"]
                    }
                    datasets_to_cache = priority_datasets.get(priority_level, priority_datasets["medium"])
                    result = await self._preprocess_and_cache_datasets(datasets_to_cache)
                
                return json.dumps({
                    "action": action,
                    "result": result,
                    "priority_level": priority_level
                })
            
            elif action == "get_stats":
                stats = {
                    "total_cached_datasets": len([k for k in self.hf_datasets_cache.keys() if not k.endswith("_processed_sample")]),
                    "processed_samples": len([k for k in self.hf_datasets_cache.keys() if k.endswith("_processed_sample")]),
                    "cache_size_mb": sum([len(str(v)) for v in self.hf_datasets_cache.values()]) / (1024 * 1024),
                    "cached_datasets": list(set([k.replace("_processed_sample", "") for k in self.hf_datasets_cache.keys()]))
                }
                
                return json.dumps({
                    "action": action,
                    "statistics": stats
                })
            
            elif action == "refresh_cache":
                # Clear existing cache
                if dataset_names:
                    for dataset_name in dataset_names:
                        self.hf_datasets_cache.pop(dataset_name, None)
                        self.hf_datasets_cache.pop(f"{dataset_name}_processed_sample", None)
                    
                    # Re-cache specified datasets
                    result = await self._preprocess_and_cache_datasets(dataset_names)
                else:
                    # Clear all and re-cache defaults
                    self.hf_datasets_cache.clear()
                    result = await self._preprocess_and_cache_datasets()
                
                return json.dumps({
                    "action": action,
                    "refreshed_datasets": dataset_names or "all",
                    "result": result
                })
            
            elif action == "list_available":
                available_datasets = [
                    "Mahesh2841/Agriculture",
                    "Dharine/agriculture-10k"
                ]
                
                return json.dumps({
                    "action": action,
                    "available_datasets": available_datasets,
                    "currently_cached": list(set([k.replace("_processed_sample", "") for k in self.hf_datasets_cache.keys()]))
                })
            
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "available_actions": ["cache_datasets", "get_stats", "refresh_cache", "list_available"]
                })
                
        except Exception as e:
            logger.error(f"❌ Dataset management failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "action": action
            })

    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health status of knowledge agent
        
        Returns:
            Dictionary with health check results
        """
        checks = {
            "openai_api": False,
            "openai_agent": False,
            "chromadb_connection": False,
            "postgres_connection": False,
            "embedding_model": False,
            "huggingface_api": False,
            "opencage_geocoding": False,
            "weather_api": False,
            "langchain_embeddings": False,
            "langchain_vectorstore": False,
            "langchain_retrieval_qa": False
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
            
            # Test ChromaDB connection
            if self.chroma_client:
                try:
                    collection = self.chroma_client.get_collection("agricultural_knowledge")
                    checks["chromadb_connection"] = True
                except:
                    checks["chromadb_connection"] = False
            
            # Test PostgreSQL connection
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                checks["postgres_connection"] = True
            
            # Test embedding model
            embedding_response = await self.client.embeddings.create(
                input="test",
                model=self.embedding_model
            )
            if embedding_response.data:
                checks["embedding_model"] = True
            
            # Test Hugging Face API
            if self.hf_api is not None:
                checks["huggingface_api"] = True
            
            # Test OpenCage Geocoding API
            try:
                test_coordinates = await self._geocode_location_with_opencage("New York, NY")
                if test_coordinates:
                    checks["opencage_geocoding"] = True
            except Exception as e:
                logger.warning(f"⚠️ OpenCage geocoding test failed: {str(e)}")
                checks["opencage_geocoding"] = False
            
            # Test Weather API
            try:
                if 'test_coordinates' in locals() and test_coordinates:
                    weather_data = await self._get_weather_data(test_coordinates[0], test_coordinates[1], 1)
                    if weather_data and "current" in weather_data:
                        checks["weather_api"] = True
            except Exception as e:
                logger.warning(f"⚠️ Weather API test failed: {str(e)}")
                checks["weather_api"] = False
            
            # Test LangChain components
            if self.langchain_embeddings is not None:
                checks["langchain_embeddings"] = True
            
            if self.langchain_vectorstore is not None:
                checks["langchain_vectorstore"] = True
            
            if self.retrieval_qa_chain is not None:
                checks["langchain_retrieval_qa"] = True
            
        except Exception as e:
            logger.error(f"❌ Knowledge agent health check failed: {str(e)}")
        
        return checks
    
    async def cleanup(self):
        """
        Cleanup resources
        """
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        if self.chroma_client:
            # ChromaDB client doesn't require explicit cleanup
            pass
        
        logger.info("🧹 Knowledge Agent cleanup completed")

