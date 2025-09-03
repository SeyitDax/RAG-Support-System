# RAG + n8n Customer Support Automation System

A production-ready AI-powered customer support system that automatically answers customer queries using RAG (Retrieval-Augmented Generation) with confidence-based routing to human agents.

## Project Overview

**Business Objective**: Reduce customer response time by 75% and achieve 60% automation rate while maintaining high customer satisfaction through intelligent AI assistance with human escalation.

**Technology Stack**:
- **AI/ML**: OpenAI GPT-4 + text-embedding-ada-002 for response generation
- **Vector Database**: Pinecone for scalable similarity search
- **Backend**: Python Flask with comprehensive API endpoints
- **Automation**: n8n workflows for intelligent routing and notifications
- **Frontend**: Professional HTML/CSS/JS demo interface

## Key Features

### **Production RAG Engine**
- OpenAI GPT-4 integration for natural language responses
- Pinecone vector database for scalable document retrieval
- Advanced confidence scoring (similarity + consistency + relevance)
- Intelligent chunking with metadata preservation
- Source attribution for transparency

### **Business Logic Automation**
- **High confidence (>0.8)**: Automatic response
- **Medium confidence (0.6-0.8)**: Review required  
- **Low confidence (<0.6)**: Escalate to human agent
- Conversation history tracking and analytics

### **Comprehensive Knowledge Base**
- **FAQ**: 30+ common customer questions and answers
- **Return Policy**: Complete return/exchange procedures
- **Shipping Info**: Domestic/international shipping options
- **Product Catalog**: Electronics, home goods, fashion, sports

### **Production API**
- RESTful endpoints with full validation
- Rate limiting and security measures  
- Comprehensive error handling and logging
- Real-time health monitoring and metrics

### **Professional Demo Interface**
- Real-time chat with confidence indicators
- Source attribution display
- Performance metrics visualization
- Mobile-responsive design

## Quick Start Guide

### Prerequisites
- Python 3.8+
- **OpenAI API key** - Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Pinecone API key** - Get from [app.pinecone.io](https://app.pinecone.io/) (free tier available)
- Git

### **Pinecone Free Tier Setup**
This system is optimized for Pinecone's free tier:

**Free Tier Includes:**
- 5 serverless indexes in AWS us-east-1 region
- Up to 2GB storage
- No credit card required

**Account Setup:**
1. Sign up at [app.pinecone.io](https://app.pinecone.io/)
2. Create an API key from the dashboard
3. **Important**: Free tier only supports AWS us-east-1 region (automatically configured)

**Common Issues:**
- "Region not supported by free plan" → System is pre-configured for free tier
- "Authentication failed" → Check your API key in .env file
- "Quota exceeded" → Free tier allows 5 indexes max

### 1. Clone and Setup
```bash
git clone <repository-url>
cd rag-support-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies (includes official pinecone v5.3.0 package)
pip install -r requirements.txt
```

### **Important Package Update**
This system now uses the **official `pinecone` package (v5.3.0)** instead of the deprecated `pinecone-client`. This ensures:
- Latest features and bug fixes
- Official support and documentation
- Better free tier compatibility
- Future-proof implementation

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
# OPENAI_API_KEY=sk-your-openai-key-here
# PINECONE_API_KEY=your-pinecone-key-here
# PINECONE_INDEX_NAME=rag-support-system
```

### 3. **Test Your Setup (Recommended)**
Before running the main demo, test your configuration:

```bash
# Test complete system (all components)
python test_system.py

# Test individual components (optional)
python test_openai.py      # OpenAI connectivity only
python test_pinecone.py    # Pinecone connectivity only
```

**Expected Output:**
```
ALL TESTS PASSED! The system should work correctly.
You can now run: python run_demo.py
```

If any tests fail, they'll provide specific guidance on how to fix the issues.

### 3. Initialize Knowledge Base
```bash
# Run the ingestion script to populate the vector database
python -c "
from src.rag_engine import RAGEngine
engine = RAGEngine()
result = engine.ingest_directory('data/knowledge_base')
print(f'Ingested {result[\"total_chunks\"]} chunks from knowledge base')
"
```

### 4. Start the API Server
```bash
# Start Flask development server (port 8000)
export FLASK_APP=src.api.app:create_app  # or src.api.app:app if not using a factory
flask run --host 0.0.0.0 --port 8000

# Health check:
# http://localhost:8000/api/health
### 5. Open Demo Interface
```bash
# Open the demo interface in your browser
# File: frontend/index.html
# Or serve with any HTTP server:
cd frontend && python -m http.server 8080
```

## API Endpoints

### Core Endpoints
- **POST** `/api/query` - Process customer questions with confidence scoring
- **POST** `/api/feedback` - Submit feedback on responses
- **GET** `/api/health` - System health check
- **GET** `/api/analytics` - Performance metrics and analytics
- **GET** `/api/system/stats` - Detailed system statistics

### Request/Response Examples

#### Query Processing
```bash
curl -X POST http://localhost:8000/api/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "How do I return a defective product?",
    "top_k": 3,
    "user_id": "demo_user"
  }'
```

**Response:**
```json
{
  "success": true,
  "response": "To return a defective product, you can initiate a return within 30 days...",
  "confidence": 0.89,
  "sources": [
    {
      "source": "return_policy.md",
      "document_type": "return_policy", 
      "relevance_score": 0.92
    }
  ],
  "should_escalate": false,
  "auto_response": true,
  "processing_time": 1.34
}
```

## Performance Metrics

### Target Business KPIs
- **Response Time Reduction**: 75% (5-10min → 1-2min)
- **Automation Rate**: 60% of queries auto-resolved
- **Customer Satisfaction**: >90% positive feedback
- **Cost Savings**: $8,000/month operational efficiency

### Technical Performance
- **Response Time**: <3 seconds average
- **Retrieval Accuracy**: >80% relevance score
- **System Uptime**: >99% availability
- **API Cost Efficiency**: <$100/month for demo usage

## System Architecture

```
Customer Query → Flask API → RAG Engine → Confidence Analysis
                     ↓
[OpenAI Embeddings] ← → [Pinecone Vector DB] ← → [GPT-4 Generation]
                     ↓
Auto-Response (>0.8) OR Human Escalation (<0.6) → n8n Workflows
                     ↓
        PostgreSQL (Analytics) + Monitoring + Notifications
```

## Development and Testing

### Running Tests
```bash
# Run unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rag_engine.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Sort imports
isort src/ tests/
```

### Development Server
```bash
## n8n Integration

The system includes pre-configured n8n workflows for:

1. **Ticket Intake**: Webhook → RAG Query → Response/Escalation
2. **Smart Routing**: Confidence-based routing logic
3. **Notifications**: Slack/Email alerts for escalated tickets
4. **Analytics**: Performance metrics collection

### n8n Setup
1. Install n8n: `npm install n8n -g`
2. Start n8n: `n8n start`
3. Import workflows from `n8n-workflows/` directory
4. Configure webhook URLs to point to your API endpoints

## Analytics and Monitoring

### Available Metrics
- Total queries processed
- Average confidence scores
- Response time distribution  
- Automation vs escalation rates
- Source document usage
- User satisfaction ratings

### Monitoring Endpoints
- `/api/health` - Basic health check
- `/api/system/stats` - Detailed system statistics  
- `/api/analytics` - Historical performance data

## Security and Production

### Security Features
- Input sanitization and validation
- Rate limiting per endpoint
- CORS configuration for frontend
- Structured logging for audit trails
- Error handling without information leakage

### Production Deployment
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 "src.api.app:create_app()"

# Or use Docker (Dockerfile included)
docker build -t rag-support-system .
docker run -p 8000:8000 --env-file .env rag-support-system
```

### Environment Variables
All configuration through environment variables:
- API keys (OpenAI, Pinecone) 
- Database connections
- Rate limiting settings
- Business logic thresholds

## Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run quality checks: `black`, `flake8`, `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request

### Code Standards
- Python 3.8+ with type hints
- Black code formatting
- Comprehensive docstrings
- Unit test coverage >80%
- Structured logging throughout

## Troubleshooting

### Common Pinecone Issues

**Error: "Region not supported by free plan"**
```bash
# Solution: The system is configured for free tier (us-east-1)
# 1. Restart the Python script completely to reload updated code
# 2. Verify your Pinecone API key is correct
# 3. Ensure you have a valid free tier account
```

**Error: "Authentication failed" or "Invalid API key"**
```bash
# Solution:
# 1. Check PINECONE_API_KEY in your .env file
# 2. Get your API key from https://app.pinecone.io/
# 3. Ensure no extra spaces or characters in the key
```

**Error: "Quota exceeded" or "Index limit reached"**
```bash
# Solution:
# 1. Free tier allows 5 indexes maximum
# 2. Check your Pinecone dashboard: https://app.pinecone.io/
# 3. Delete unused indexes if needed
```

### Common OpenAI Issues

**Error: "OpenAI API key must be provided"**
```bash
# Solution:
# 1. Check OPENAI_API_KEY in your .env file  
# 2. Get your API key from https://platform.openai.com/api-keys
# 3. Ensure you have sufficient OpenAI credits
```

### General Issues

**Knowledge base initialization fails:**
```bash
# Solution:
# 1. Restart the script completely (close terminal, reopen)
# 2. Check both API keys are valid
# 3. Verify internet connection
# 4. Continue anyway - you can ingest documents later via /api/ingest
```

**Script shows old error messages:**
```bash
# Solution: Python module caching issue
# 1. Close all Python processes completely
# 2. Restart terminal/command prompt
# 3. Run the script again
```

## Support and Contact

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Issues**: Open GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions

### Portfolio Context
This project demonstrates:
- **AI Integration Expertise**: Production RAG implementation
- **Full-Stack Development**: Backend API + Frontend UI
- **Business Logic**: Confidence-based automation decisions
- **DevOps Practices**: Testing, monitoring, deployment readiness
- **Clear Value Proposition**: Measurable business impact

**Target Market**: $35-50/hour AI Integration Specialist for enterprise customers needing intelligent customer support automation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready for Upwork Portfolio**: This system demonstrates production-ready AI integration capabilities with clear business value, comprehensive documentation, and measurable performance targets that justify premium consulting rates.