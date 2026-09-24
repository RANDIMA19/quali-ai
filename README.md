# Quality AI Diagnostic System

An AI-powered incident diagnosis system for textile manufacturing quality assurance.

## System Overview

This system uses a multi-agent pipeline to analyze production incidents and provide root-cause diagnosis:

1. **Agent 1 (Intake)**: Extracts entities from natural language queries
2. **Agent 2 (Anomaly)**: Detects statistical anomalies in production data
3. **Agent 3 (Retrieval)**: Retrieves similar historical incidents
4. **Agent 4 (Synthesis)**: Synthesizes diagnosis using rule-based logic

## Architecture

```
User Interface (Streamlit)
        ↓
FastAPI Backend (JWT Auth)
        ↓
Orchestrator Pipeline
        ↓
Multi-Agent System
```

## Features

- **AI-Powered Diagnosis**: Multi-agent pipeline for incident analysis
- **Historical Context**: Retrieves and cites similar past incidents
- **Anomaly Detection**: Statistical analysis of production data
- **Secure API**: JWT authentication with role-based access control
- **Web Interface**: User-friendly chat interface
- **Input Sanitization**: Protection against injection attacks

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running the System

#### Option 1: Automatic Startup (Recommended)

**Windows:**
```bash
start_system.bat
```

**Linux/Mac:**
```bash
chmod +x start_system.sh
./start_system.sh
```

#### Option 2: Manual Startup

**Terminal 1 - Start API:**
```bash
python api/main.py
```

**Terminal 2 - Start UI:**
```bash
streamlit run ui/app.py
```

### Access Points

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **UI**: http://localhost:8501

## Demo Credentials

- **qa_officer** / `qa123`
- **technician** / `tech123`
- **manager** / `mgr123`

## Project Structure

```
quali-ai/
├── agents/                 # Multi-agent system
│   ├── agent1_intake.py   # Entity extraction
│   ├── agent2_anomaly.py  # Anomaly detection
│   ├── agent3_retrieval.py # Historical retrieval
│   ├── agent4_synthesis.py # LLM-based synthesis (original)
│   ├── agent4_synthesis_mock.py # Rule-based synthesis (current)
│   └── orchestrator.py    # Pipeline coordination
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints and auth
│   └── README.md          # API documentation
├── ui/                     # Streamlit frontend
│   ├── app.py             # Web interface
│   ├── .streamlit/        # Streamlit config
│   └── README.md          # UI documentation
├── data/                   # Data files
├── test_agents.py         # Agent testing script
├── test_api.py            # API testing script
├── check_ollama.py        # Ollama diagnostic tool
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
└── README.md             # This file
```

## Configuration

Edit `.env` file to configure:

```env
# Ollama Configuration (if using LLM-based Agent 4)
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=mistral

# JWT Configuration
JWT_SECRET_KEY=your-secret-key-change-in-production

# API Configuration
API_BASE_URL=http://localhost:8000
```

## Usage Examples

### Via Web Interface

1. Open http://localhost:8501
2. Login with demo credentials
3. Ask questions like:
   - "Why is Loom 12 producing inconsistent elongation in Batch B102?"
   - "What's causing high tension readings on Loom 5?"
   - "Investigate the strength issues in Batch A205"

### Via API

```bash
# Get authentication token
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "qa_officer", "password": "qa123"}'

# Use token to diagnose
curl -X POST "http://localhost:8000/diagnose" \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is Loom 12 producing inconsistent elongation in Batch B102?"}'
```

### Via Python Script

```python
from agents.orchestrator import run_pipeline

result = run_pipeline(
    query="Why is Loom 12 producing inconsistent elongation in Batch B102?",
    production_df=None,  # Optional: DataFrame with production data
    metric_name=None,     # Optional: Metric to analyze
    top_k=3              # Number of similar incidents to retrieve
)

print(result)
```

## Testing

### Test Individual Agents

```bash
python test_agents.py
```

### Test API Endpoints

```bash
python test_api.py
```

### Check Ollama (if using LLM-based Agent 4)

```bash
python check_ollama.py
```

## Security Features

- **Authentication**: JWT token-based authentication
- **Authorization**: Role-based access control (qa_officer, technician, manager)
- **Input Sanitization**: SQL injection and XSS prevention
- **Password Hashing**: SHA-256 password hashing
- **CORS Protection**: Configurable CORS middleware

## Agent Details

### Agent 1: Entity Extraction
- Uses spaCy NLP for entity extraction
- Extracts: loom_id, batch_id, metric, defect_symptom
- Processes natural language queries

### Agent 2: Anomaly Detection
- Statistical analysis using Z-score method
- Configurable threshold for anomaly detection
- Correlation analysis between metrics

### Agent 3: Historical Retrieval
- Uses sentence-transformers for semantic search
- ChromaDB for vector storage
- Retrieves similar past incidents with similarity scores

### Agent 4: Synthesis (Current: Rule-based)
- Rule-based logic for diagnosis generation
- Determines primary cause based on patterns
- Generates prioritized recommendations
- Cites relevant historical incidents

## Troubleshooting

### Port Already in Use
If ports 8000 or 8501 are already in use:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### SpaCy Model Missing
```bash
python -m spacy download en_core_web_sm
```

### API Connection Issues
1. Ensure FastAPI backend is running
2. Check API_BASE_URL in .env file
3. Verify firewall settings

### Agent 4 Not Working
The system currently uses rule-based Agent 4 (no external dependencies). To use LLM-based Agent 4:
1. Install Ollama from https://ollama.com
2. Run: `ollama pull mistral`
3. Ensure Ollama service is running
4. Update orchestrator to use original Agent 4

## Development

### Adding New Agents
1. Create new agent file in `agents/` directory
2. Implement agent function following existing patterns
3. Update `orchestrator.py` to include new agent
4. Add tests in `test_agents.py`

### Extending the API
1. Add new endpoints in `api/main.py`
2. Update authentication/authorization as needed
3. Add corresponding tests in `test_api.py`
4. Update API documentation

### Customizing the UI
1. Modify `ui/app.py` for functionality changes
2. Update `ui/.streamlit/config.toml` for appearance
3. Add new components following existing patterns

## License

This project is for demonstration purposes.

## Support

For issues and questions, please refer to the individual README files:
- API: `api/README.md`
- UI: `ui/README.md`
