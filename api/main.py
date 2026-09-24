"""
FastAPI application for the Quality AI Diagnostic System
Provides JWT authentication, role-based access control, and input sanitization
"""

import os
import re
import json
import hashlib
from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from jose import JWTError, jwt
import sys

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import run_pipeline

# ── Configuration ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Quality AI Diagnostic API",
    description="AI-powered incident diagnosis system for textile manufacturing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security scheme
security = HTTPBearer()

# Simple password hashing (SHA-256)
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password

# ── User Database (In-memory for demo - use proper database in production) ───────
users_db = {
    "qa_officer": {
        "username": "qa_officer",
        "hashed_password": hash_password("qa123"),
        "roles": ["qa_officer"]
    },
    "technician": {
        "username": "technician",
        "hashed_password": hash_password("tech123"),
        "roles": ["technician"]
    },
    "manager": {
        "username": "manager",
        "hashed_password": hash_password("mgr123"),
        "roles": ["manager"]
    }
}

# ── Pydantic Models ─────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str
    roles: List[str]

class TokenData(BaseModel):
    username: Optional[str] = None
    roles: List[str] = []

class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class DiagnoseRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500, description="Natural language query about the incident")
    
    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize input to prevent injection attacks"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';]', '', v)
        
        # Limit length
        if len(sanitized) > 500:
            raise ValueError("Query too long")
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'(union|select|insert|delete|update|drop|alter|create|exec)',
            r'(--|;|/\*|\*/|@@)',
            r'(or|and)\s+\d+\s*=\s*\d+',
            r'(or|and)\s+["\'].*["\']\s*=\s*["\'].*["\']'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError("Query contains potentially malicious content")
        
        return sanitized.strip()

class DiagnoseResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    timestamp: datetime

# ── Authentication Functions ─────────────────────────────────────────────────────

def get_user(username: str) -> Optional[dict]:
    """Get user from database"""
    return users_db.get(username)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate user credentials"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Verify JWT token and extract user data"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        roles: List[str] = payload.get("roles", [])
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username, roles=roles)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data

def require_roles(allowed_roles: List[str]):
    """Dependency factory to require specific roles"""
    def role_checker(token_data: TokenData = Depends(verify_token)) -> TokenData:
        user_roles = token_data.roles
        if not any(role in allowed_roles for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}",
            )
        return token_data
    return role_checker

# ── API Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quality AI Diagnostic API",
        "version": "1.0.0",
        "endpoints": {
            "token": "/token",
            "diagnose": "/diagnose",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "Quality AI Diagnostic System"
    }

@app.post("/token", response_model=Token)
async def login(user_login: UserLogin):
    """
    Authenticate user and return JWT token
    
    Demo credentials:
    - qa_officer / qa123
    - technician / tech123  
    - manager / mgr123
    """
    user = authenticate_user(user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "roles": user["roles"]},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        roles=user["roles"]
    )

@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose_incident(
    request: DiagnoseRequest,
    token_data: TokenData = Depends(require_roles(["qa_officer", "technician", "manager"]))
):
    """
    Run diagnostic pipeline on incident query
    
    Requires authentication with one of these roles: qa_officer, technician, manager
    """
    try:
        # Log the request
        print(f"[{datetime.utcnow()}] Diagnosis request from user: {token_data.username}")
        print(f"[{datetime.utcnow()}] User roles: {token_data.roles}")
        print(f"[{datetime.utcnow()}] Query: {request.query}")
        
        # Run the orchestrator pipeline
        result = run_pipeline(
            query=request.query,
            production_df=None,  # Can be extended to accept production data
            metric_name=None,
            top_k=3
        )
        
        # Check for errors in the pipeline
        agent4_output = result.get("agent_outputs", {}).get("agent4_synthesis", {})
        if "error" in agent4_output:
            return DiagnoseResponse(
                success=False,
                message=f"Pipeline error: {agent4_output['error']}",
                data=result,
                timestamp=datetime.utcnow()
            )
        
        return DiagnoseResponse(
            success=True,
            message="Diagnosis completed successfully",
            data=result,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        print(f"[{datetime.utcnow()}] Error in diagnosis: {str(e)}")
        return DiagnoseResponse(
            success=False,
            message=f"Internal server error: {str(e)}",
            data=None,
            timestamp=datetime.utcnow()
        )

@app.get("/users/me")
async def read_users_me(token_data: TokenData = Depends(verify_token)):
    """Get current user information"""
    user = get_user(token_data.username)
    if user:
        return {
            "username": user["username"],
            "roles": user["roles"]
        }
    raise HTTPException(status_code=404, detail="User not found")

# ── Role-Protected Examples ──────────────────────────────────────────────────────

@app.get("/admin/stats")
async def get_admin_stats(
    token_data: TokenData = Depends(require_roles(["manager"]))
):
    """Admin-only endpoint - only managers can access"""
    return {
        "message": "Admin statistics",
        "total_users": len(users_db),
        "system_status": "operational"
    }

@app.get("/tech/equipment")
async def get_equipment_info(
    token_data: TokenData = Depends(require_roles(["technician", "manager"]))
):
    """Technician and manager endpoint"""
    return {
        "message": "Equipment information",
        "looms": ["Loom 1", "Loom 2", "Loom 12"],
        "status": "active"
    }

# ── Main Application Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
