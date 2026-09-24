# Quality AI Diagnostic UI

Streamlit-based user interface for the Quality AI Diagnostic System.

## Features

- **User Authentication**: Secure login with role-based access
- **Chat Interface**: Interactive chat for diagnostic questions
- **Rich Response Display**: Formatted display of diagnosis results
- **Historical Incident Citations**: Shows relevant past incidents
- **Recommendations**: Actionable recommendations with priority levels
- **Session Management**: Chat history and user session handling

## Installation

```bash
pip install -r ../requirements.txt
```

## Running the UI

Make sure the FastAPI backend is running first, then:

```bash
streamlit run ui/app.py
```

The UI will be available at `http://localhost:8501`

## Configuration

Environment variables (optional):

```env
API_BASE_URL=http://localhost:8000
```

## Usage

1. **Login**: Use one of the demo credentials:
   - `qa_officer` / `qa123`
   - `technician` / `tech123`
   - `manager` / `mgr123`

2. **Ask Questions**: Type diagnostic questions in the chat interface:
   - "Why is Loom 12 producing inconsistent elongation in Batch B102?"
   - "What's causing high tension readings on Loom 5?"
   - "Investigate the strength issues in Batch A205"

3. **View Results**: The system will display:
   - Extracted entity information
   - Root cause analysis with severity
   - Contributing factors
   - Cited historical incidents
   - Prioritized recommendations
   - Confidence levels

## Features

### Authentication
- Secure login against FastAPI backend
- JWT token-based session management
- Role-based access control
- Automatic session timeout handling

### Chat Interface
- Real-time chat with the diagnostic system
- Chat history persistence during session
- Clear history functionality
- Responsive design

### Response Display
- **Entity Extraction**: Shows parsed loom ID, batch ID, metric, and symptoms
- **Root Cause Analysis**: Primary cause, severity, and contributing factors
- **Historical Incidents**: Similar past incidents with relevance scores
- **Recommendations**: Actionable steps with priority indicators
- **Confidence Levels**: System confidence in the diagnosis

## Troubleshooting

### Connection Issues
If you see connection errors:
1. Ensure the FastAPI backend is running on `http://localhost:8000`
2. Check that the API endpoint is accessible
3. Verify the `API_BASE_URL` environment variable if using a different endpoint

### Authentication Issues
If login fails:
1. Verify the backend is running
2. Check the demo credentials
3. Ensure your user role has permission to access the `/diagnose` endpoint

### Slow Responses
The diagnostic pipeline may take several seconds due to:
- AI processing time
- Historical incident retrieval
- Anomaly detection calculations

## Development

The UI communicates with the FastAPI backend using REST API calls. The main integration points are:

- `/token` - User authentication
- `/diagnose` - Diagnostic queries
- JWT token handling for authenticated requests

## Future Enhancements

- Production data upload for anomaly detection
- Real-time dashboard for production metrics
- Incident report generation
- Multi-language support
- Advanced filtering and search
