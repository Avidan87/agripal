# 🌾 AgriPal - AI-Powered Agricultural Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An intelligent agricultural assistant system powered by OpenAI Agents SDK, featuring multi-agent orchestration for crop analysis, knowledge retrieval, and automated reporting.**

## 🎯 Overview

AgriPal is a comprehensive AI-powered agricultural assistant that combines computer vision, natural language processing, and intelligent agent coordination to provide farmers with actionable insights for crop management, disease detection, and farming optimization.

### Key Features

- **🔍 Visual Crop Analysis**: Advanced image analysis using GPT-4o Vision for disease detection, pest identification, and crop health assessment
- **📚 Intelligent Knowledge Retrieval**: RAG-powered agricultural knowledge base with semantic search and contextual recommendations
- **📧 Automated Report Generation**: AI-generated consultation reports with professional email distribution
- **🤖 Multi-Agent Orchestration**: Intelligent workflow coordination using OpenAI Agents SDK
- **🌐 Modern Web Interface**: Responsive Gradio-based frontend with real-time chat capabilities
- **🗄️ Robust Data Architecture**: PostgreSQL + ChromaDB vector database for scalable knowledge management

## 🏗️ Architecture

### System Components

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Gradio Web Interface]
        Chat[Real-time Chat]
        Upload[Image Upload]
    end
    
    subgraph "API Layer"
        FastAPI[FastAPI Backend]
        Auth[Authentication]
        Middleware[Rate Limiting]
    end
    
    subgraph "AI Agent Layer"
        Coordinator[Agent Coordinator]
        Perception[Perception Agent]
        Knowledge[Knowledge Agent]
        Email[Email Agent]
    end
    
    subgraph "Data Layer"
        PostgreSQL[(PostgreSQL)]
        ChromaDB[(ChromaDB)]
        Redis[(Redis Cache)]
    end
    
    subgraph "External Services"
        OpenAI[OpenAI API]
        SendGrid[SendGrid Email]
        Weather[Weather API]
        Geocoding[OpenCage API]
    end
    
    UI --> FastAPI
    FastAPI --> Coordinator
    Coordinator --> Perception
    Coordinator --> Knowledge
    Coordinator --> Email
    Perception --> OpenAI
    Knowledge --> ChromaDB
    Knowledge --> PostgreSQL
    Email --> SendGrid
    FastAPI --> PostgreSQL
    FastAPI --> Redis
```

### AI Agent Architecture

#### 🎯 Agent Coordinator
- **Purpose**: Central orchestration hub using OpenAI Agents SDK
- **Capabilities**: 
  - Intelligent workflow decision-making
  - Dynamic agent selection based on user input
  - Error recovery and fallback strategies
  - Performance optimization and monitoring

#### 🔍 Perception Agent
- **Purpose**: Visual analysis and crop health assessment
- **Capabilities**:
  - Disease detection and classification using GPT-4o Vision
  - Pest identification and management recommendations
  - Crop stress assessment (drought, nutrient deficiency)
  - Growth stage analysis and yield prediction
  - Image quality validation and enhancement

#### 📚 Knowledge Agent
- **Purpose**: RAG-powered agricultural knowledge retrieval
- **Capabilities**:
  - Semantic search through agricultural knowledge base
  - Contextual information retrieval from PostgreSQL
  - Weather data integration for localized recommendations
  - Agricultural best practices and research synthesis
  - Crop-specific and region-specific advice

#### 📧 Email Agent
- **Purpose**: Intelligent report generation and distribution
- **Capabilities**:
  - Session summary generation using AI
  - Professional agricultural report formatting
  - Email delivery via SendGrid
  - Template-based report generation
  - Follow-up scheduling and recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- OpenAI API Key
- SendGrid API Key (optional)
- Weather API Key (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/agripal.git
cd agripal
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Initialize database**
```bash
python scripts/setup_database.py
```

6. **Start the backend**
```bash
python run_backend.py
```

7. **Start the frontend** (in a new terminal)
```bash
cd frontend
python main.py
```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/agripal
REDIS_URL=redis://localhost:6379

# Email Configuration (Optional)
SENDGRID_API_KEY=your_sendgrid_api_key_here
SENDGRID_FROM_EMAIL=noreply@agripal.com

# Weather API (Optional)
WEATHER_API_KEY=your_weather_api_key_here
OPENCAGE_API_KEY=your_opencage_api_key_here

# Application Settings
ENVIRONMENT=development
DEBUG=True
```

## 📖 Usage

### Web Interface

1. **Access the application**: Navigate to `http://localhost:7860`
2. **Upload crop images**: Use the image upload feature for visual analysis
3. **Ask questions**: Type agricultural questions in the chat interface
4. **Get recommendations**: Receive AI-powered farming advice and reports

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/api/v1/agents/health
```

#### Full Consultation
```bash
curl -X POST "http://localhost:8000/api/v1/agents/consultation/full" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My corn plants have yellow spots on the leaves",
    "crop_type": "corn",
    "location": "California, USA"
  }'
```

#### Image Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/agents/workflows/perception-to-knowledge" \
  -F "query=Analyze this crop disease" \
  -F "files=@crop_image.jpg"
```

## 🗄️ Database Schema

### Core Tables

- **`users`**: Farmer profiles with location and farm details
- **`chat_sessions`**: Individual consultation sessions
- **`session_messages`**: Messages within sessions
- **`analysis_reports`**: AI analysis results and recommendations
- **`knowledge_documents`**: Agricultural knowledge base documents
- **`rag_query_analytics`**: RAG query performance analytics

### Vector Database

- **ChromaDB**: Vector embeddings for semantic search
- **Collection**: `agricultural_knowledge`
- **Embedding Model**: `text-embedding-3-large`

## 🔧 Development

### Project Structure

```
agripal/
├── backend/
│   ├── agripal_agents/          # AI Agent implementations
│   │   ├── perception_agent.py  # Visual analysis agent
│   │   ├── knowledge_agent.py    # RAG knowledge agent
│   │   └── email_agent.py        # Report generation agent
│   ├── coordinator/              # Agent orchestration
│   │   ├── agent_coordinator.py # Main coordinator
│   │   └── workflow_models.py    # Workflow definitions
│   ├── api/                      # FastAPI routes
│   ├── database/                 # Database models and services
│   ├── middleware/               # Authentication and rate limiting
│   └── models.py                 # Pydantic data models
├── frontend/
│   ├── agripal_ui.py            # Gradio interface
│   └── main.py                  # Frontend entry point
├── scripts/                     # Database setup scripts
├── chroma_db/                   # ChromaDB vector storage
└── requirements.txt             # Python dependencies
```

### Key Technologies

- **Backend**: FastAPI, SQLAlchemy, AsyncPG
- **AI/ML**: OpenAI Agents SDK, LangChain, ChromaDB
- **Frontend**: Gradio, HTML/CSS/JavaScript
- **Database**: PostgreSQL, Redis
- **Email**: SendGrid, Jinja2 templates
- **Authentication**: JWT, HTTPBearer

### Adding New Agents

1. **Create agent class** in `backend/agripal_agents/`
2. **Implement required methods**: `__init__()`, `process()`, `health_check()`
3. **Register with coordinator** in `agent_coordinator.py`
4. **Add API endpoints** in `backend/api/agent_routes.py`

### Customizing Workflows

Workflows are defined in `backend/coordinator/workflow_models.py`:

```python
class WorkflowType(Enum):
    FULL_CONSULTATION = "full_consultation"
    IMAGE_ANALYSIS_ONLY = "image_analysis_only"
    KNOWLEDGE_SEARCH_ONLY = "knowledge_search_only"
    EMAIL_REPORT_ONLY = "email_report_only"
```

## 📊 Performance & Monitoring

### Metrics Tracking

- **Workflow Performance**: Execution time, success rates
- **Agent Health**: Individual agent status and capabilities
- **RAG Analytics**: Query performance, retrieval quality
- **User Engagement**: Session duration, message counts

### Health Checks

```bash
# Check system health
curl http://localhost:8000/api/v1/agents/health

# Get performance metrics
curl http://localhost:8000/api/v1/agents/metrics
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=backend --cov-report=html
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Production Considerations

- **Environment Variables**: Secure API key management
- **Database**: PostgreSQL with connection pooling
- **Caching**: Redis for session and rate limiting
- **Monitoring**: Application performance monitoring
- **Security**: HTTPS, authentication, rate limiting

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include comprehensive docstrings
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the powerful GPT-4o and Agents SDK
- **FastAPI** for the excellent web framework
- **Gradio** for the intuitive interface framework
- **LangChain** for RAG implementation
- **ChromaDB** for vector database capabilities

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-org/agripal/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/agripal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/agripal/discussions)
- **Email**: support@agripal.com

---

**Built with ❤️ for the agricultural community**

*Empowering farmers with AI-driven insights for sustainable and productive agriculture.*
