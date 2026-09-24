"""
Streamlit UI for Quality AI Diagnostic System
Simple chat interface with login, diagnostic questions, and root-cause answers
"""

import streamlit as st
import requests
import json
from datetime import datetime
import os

# ── Configuration ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quality AI Diagnostic System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .incident-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2ecc71;
    }
    .error-message {
        background-color: #fee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
    }
    .success-message {
        background-color: #efe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Management ─────────────────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_roles' not in st.session_state:
    st.session_state.user_roles = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ── API Functions ────────────────────────────────────────────────────────────────

def login_user(username: str, password: str) -> bool:
    """Authenticate user with the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/token",
            json={"username": username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            token_data = response.json()
            st.session_state.token = token_data["access_token"]
            st.session_state.username = username
            st.session_state.user_roles = token_data["roles"]
            st.session_state.authenticated = True
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return False

def get_diagnosis(query: str) -> dict:
    """Get diagnosis from the API"""
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/diagnose",
            headers=headers,
            json={"query": query},
            timeout=60  # Longer timeout for AI processing
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.session_state.authenticated = False
            st.session_state.token = None
            st.error("Session expired. Please login again.")
            return None
        else:
            st.error(f"API error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def logout_user():
    """Logout user and clear session"""
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.user_roles = []
    st.session_state.chat_history = []

# ── UI Components ───────────────────────────────────────────────────────────────

def render_login_page():
    """Render login page"""
    st.markdown('<div class="main-header">🔬 Quality AI Diagnostic System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if not username or not password:
                    st.warning("Please enter both username and password")
                else:
                    if login_user(username, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        st.markdown("---")
        st.info("**Demo Credentials:**")
        st.code("""
qa_officer / qa123
technician / tech123
manager / mgr123
""")

def render_chat_interface():
    """Render main chat interface"""
    # Header
    st.markdown('<div class="main-header">🔬 Quality AI Diagnostic System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.subheader("User Information")
        st.write(f"**Username:** {st.session_state.username}")
        st.write(f"**Roles:** {', '.join(st.session_state.user_roles)}")
        
        st.markdown("---")
        
        st.subheader("Actions")
        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("Chat History")
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.session_state.chat_history:
            st.write(f"Total conversations: {len(st.session_state.chat_history)}")
        
        st.markdown("---")
        st.info(f"API Endpoint: {API_BASE_URL}")
    
    # Main chat area
    st.markdown("### Diagnostic Assistant")
    st.markdown("Ask questions about production incidents, quality issues, or equipment problems.")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                if answer.get("success"):
                    render_diagnosis_response(answer["data"])
                else:
                    st.error(f"Error: {answer.get('message', 'Unknown error')}")
    
    # Chat input
    if prompt := st.chat_input("Ask a diagnostic question..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get diagnosis
        with st.chat_message("assistant"):
            with st.spinner("Analyzing incident..."):
                diagnosis = get_diagnosis(prompt)
                
                if diagnosis:
                    # Add to chat history
                    st.session_state.chat_history.append((prompt, diagnosis))
                    
                    if diagnosis.get("success"):
                        render_diagnosis_response(diagnosis["data"])
                    else:
                        st.error(f"Error: {diagnosis.get('message', 'Unknown error')}")
                else:
                    st.error("Failed to get diagnosis. Please try again.")

def render_diagnosis_response(data: dict):
    """Render the diagnosis response in a formatted way"""
    if not data:
        st.warning("No diagnosis data available")
        return
    
    # Get agent outputs
    agent_outputs = data.get("agent_outputs", {})
    
    # Agent 1: Entity Extraction
    agent1_data = agent_outputs.get("agent1_intake", {})
    if agent1_data and "error" not in agent1_data:
        st.markdown("#### 📋 Extracted Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Loom ID", agent1_data.get("loom_id", "N/A"))
        with col2:
            st.metric("Batch ID", agent1_data.get("batch_id", "N/A"))
        with col3:
            st.metric("Metric", agent1_data.get("metric", "N/A"))
        with col4:
            st.metric("Symptom", agent1_data.get("defect_symptom", "N/A"))
    
    # Agent 4: Diagnosis
    agent4_data = agent_outputs.get("agent4_synthesis", {})
    if agent4_data and "error" not in agent4_data:
        diagnosis_info = agent4_data.get("diagnosis", {})
        
        if diagnosis_info:
            # Root Cause Analysis
            rca = diagnosis_info.get("root_cause_analysis", {})
            if rca:
                st.markdown("#### 🔍 Root Cause Analysis")
                
                severity_colors = {
                    "low": "🟢",
                    "medium": "🟡", 
                    "high": "🟠",
                    "critical": "🔴"
                }
                severity_icon = severity_colors.get(rca.get("severity", "medium"), "⚪")
                
                st.markdown(f"**Primary Cause:** {rca.get('primary_cause', 'N/A')}")
                st.markdown(f"**Severity:** {severity_icon} {rca.get('severity', 'N/A').upper()}")
                
                contributing_factors = rca.get("contributing_factors", [])
                if contributing_factors:
                    st.markdown("**Contributing Factors:**")
                    for factor in contributing_factors:
                        st.markdown(f"- {factor}")
            
            # Cited Incidents
            cited_incidents = diagnosis_info.get("cited_incidents", [])
            if cited_incidents:
                st.markdown("#### 📚 Cited Historical Incidents")
                for incident in cited_incidents:
                    with st.container():
                        st.markdown(f"""
                        <div class="incident-card">
                            <strong>Incident ID:</strong> {incident.get('incident_id', 'N/A')}<br>
                            <strong>Relevance:</strong> {incident.get('relevance', 'N/A')}<br>
                            <strong>Similarity Score:</strong> {incident.get('similarity_score', 0):.3f}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Recommendations
            recommendations = diagnosis_info.get("recommendations", [])
            if recommendations:
                st.markdown("#### 💡 Recommendations")
                priority_order = {"immediate": 0, "short-term": 1, "long-term": 2}
                sorted_recommendations = sorted(recommendations, key=lambda x: priority_order.get(x.get("priority", "long-term"), 3))
                
                for rec in sorted_recommendations:
                    priority_colors = {
                        "immediate": "🔴",
                        "short-term": "🟡",
                        "long-term": "🟢"
                    }
                    priority_icon = priority_colors.get(rec.get("priority", "long-term"), "⚪")
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{priority_icon} {rec.get('action', 'N/A')}</strong><br>
                        <em>Priority: {rec.get('priority', 'N/A')}</em><br>
                        <strong>Expected Outcome:</strong> {rec.get('expected_outcome', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Confidence and Notes
            confidence = diagnosis_info.get("confidence", "N/A")
            additional_notes = diagnosis_info.get("additional_notes", "")
            
            if confidence != "N/A" or additional_notes:
                st.markdown("#### 📊 Additional Information")
                if confidence != "N/A":
                    st.markdown(f"**Confidence Level:** {confidence.upper()}")
                if additional_notes:
                    st.markdown(f"**Notes:** {additional_notes}")
    
    # Agent 3: Historical Incidents (fallback if Agent 4 doesn't have citations)
    agent3_data = agent_outputs.get("agent3_retrieval", [])
    if agent3_data and isinstance(agent3_data, list) and not diagnosis_info.get("cited_incidents"):
        st.markdown("#### 📚 Similar Historical Incidents")
        for incident in agent3_data[:3]:  # Show top 3
            with st.container():
                st.markdown(f"""
                <div class="incident-card">
                    <strong>Incident ID:</strong> {incident.get('incident_id', 'N/A')}<br>
                    <strong>Reason:</strong> {incident.get('reason', 'N/A')}<br>
                    <strong>Similarity Score:</strong> {incident.get('similarity_score', 0):.3f}
                </div>
                """, unsafe_allow_html=True)

# ── Main Application ────────────────────────────────────────────────────────────

def main():
    """Main application logic"""
    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()
