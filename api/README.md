# Quality AI Diagnostic API

FastAPI application for the Quality AI Diagnostic System with JWT authentication and role-based access control.

## Features

- **JWT Authentication**: Secure token-based authentication
- **Role-Based Access Control**: Three user roles with different permissions
- **Input Sanitization**: Protection against injection attacks
- **Incident Diagnosis**: AI-powered diagnostic pipeline integration
- **CORS Support**: Cross-origin resource sharing enabled

## Installation

```bash
pip install -r ../requirements.txt
```

## Running the API

```bash
# Development server
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Authentication

#### POST /token
Authenticate user and receive JWT token.

**Demo Credentials:**
- `qa_officer` / `qa123`
- `technician` / `tech123`
- `manager` / `mgr123`

**Request:**
```json
{
  "username": "qa_officer",
  "password": "qa123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "roles": ["qa_officer"]
}
```

### Diagnosis

#### POST /diagnose
Run diagnostic pipeline on incident query.

**Headers:**
```
Authorization: Bearer <your_token>
```

**Request:**
```json
{
  "query": "Why is Loom 12 producing inconsistent elongation in Batch B102?"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Diagnosis completed successfully",
  "data": {
    "query": "Why is Loom 12 producing inconsistent elongation in Batch B102?",
    "agent_outputs": {
      "agent1_intake": {...},
      "agent2_anomaly": {...},
      "agent3_retrieval": [...],
      "agent4_synthesis": {...}
    }
  },
  "timestamp": "2024-09-22T12:00:00"
}
```

### Role-Based Endpoints

#### GET /users/me
Get current user information (requires authentication)

#### GET /admin/stats
Admin statistics (requires `manager` role)

#### GET /tech/equipment
Equipment information (requires `technician` or `manager` role)

### System

#### GET /
API information and available endpoints

#### GET /health
Health check endpoint

## Security Features

### Input Sanitization
- Removes dangerous characters (`<>"';`)
- Length validation (10-500 characters)
- SQL injection pattern detection
- XSS prevention

### Authentication
- JWT token-based authentication
- SHA-256 password hashing
- Token expiration (30 minutes)
- Secure token verification

### Authorization
- Role-based access control
- Multiple role support per user
- Granular endpoint protection

## User Roles

### qa_officer
- Access to `/diagnose` endpoint
- Can view user information
- Cannot access admin endpoints

### technician
- Access to `/diagnose` endpoint
- Can view user information
- Can access equipment information
- Cannot access admin endpoints

### manager
- Full access to all endpoints
- Can view admin statistics
- Can access equipment information
- Can use diagnostic features

## Configuration

Environment variables (set in `.env` file):

```env
JWT_SECRET_KEY=your-secret-key-change-in-production
```

## Development

The API uses an in-memory user database for demonstration. In production, implement a proper database integration.

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `401`: Unauthorized (invalid/missing token)
- `403`: Forbidden (insufficient permissions)
- `422`: Validation error (invalid input)
- `500`: Internal server error

## Testing with curl

```bash
# Get token
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "qa_officer", "password": "qa123"}'

# Use token to diagnose
curl -X POST "http://localhost:8000/diagnose" \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is Loom 12 producing inconsistent elongation in Batch B102?"}'
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
