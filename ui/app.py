"""
Streamlit UI for Quality AI Diagnostic System
Simple chat interface with login, diagnostic questions, and root-cause answers
"""

import streamlit as st
import requests
import json
from datetime import datetime
import os
from html import escape

# ── Configuration ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

EXAMPLE_QUESTIONS = [
    "Loom 14 is producing weft bars on batch B-2207. What is the likely cause?",
    "Fabric weight on batch B-1983 is below spec. Have we seen this before?",
    "Frequent warp breaks on Loom 7 since the last beam change.",
]

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

:root {
    --ink: #14213d;
    --indigo: #2f3f8f;
    --indigo-2: #4b5fd0;
    --indigo-soft: #e8ebf7;
    --paper: #f4f6fb;
    --card: #ffffff;
    --line: #dde1ea;
    --muted: #5b6478;
    --ok: #2f7d5b;
    --warn: #b7791f;
    --hot: #c2571a;
    --crit: #b42318;
}

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.bi { vertical-align: -0.12em; }
.stApp { background: var(--paper); }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 1150px; position: relative; z-index: 1; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ── Keyframes ── */
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes floatA { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(60px,-80px) scale(1.15); } }
@keyframes floatB { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-70px,60px) scale(0.9); } }
@keyframes weave { from { background-position: 0 0, 0 0; } to { background-position: 140px 0, 0 140px; } }
@keyframes shine { to { background-position: 200% center; } }
@keyframes fillBar { from { width: 0; } }
@keyframes pulseRing { 0% { box-shadow: 0 0 0 0 rgba(180,35,24,0.55); } 100% { box-shadow: 0 0 0 10px rgba(180,35,24,0); } }
@keyframes avatarRing { 0% { box-shadow: 0 0 0 0 rgba(120,140,255,0.6); } 100% { box-shadow: 0 0 0 12px rgba(120,140,255,0); } }
@keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }

/* ── Animated login background (fixed layer behind everything) ── */
.bg-anim { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;
    background: linear-gradient(120deg, #eef1fb, #dfe6fa, #f3ecfb, #e3f1f6);
    background-size: 300% 300%; animation: gradientShift 18s ease infinite; }
.bg-anim .threads { position: absolute; inset: -140px;
    background-image:
        repeating-linear-gradient(90deg, rgba(47,63,143,0.07) 0 2px, transparent 2px 28px),
        repeating-linear-gradient(0deg, rgba(47,63,143,0.07) 0 2px, transparent 2px 28px);
    background-size: 140px 140px, 140px 140px; animation: weave 14s linear infinite; }
.bg-anim .orb { position: absolute; border-radius: 50%; filter: blur(60px); opacity: 0.55; }
.bg-anim .o1 { width: 420px; height: 420px; background: #7c8cf0; top: -80px; left: -60px; animation: floatA 16s ease-in-out infinite; }
.bg-anim .o2 { width: 360px; height: 360px; background: #5fc7d6; bottom: -80px; right: 8%; animation: floatB 19s ease-in-out infinite; }
.bg-anim .o3 { width: 300px; height: 300px; background: #c59cf0; top: 40%; right: -80px; animation: floatA 22s ease-in-out infinite reverse; }

/* ── Header ── */
.main-header {
    font-size: 2.5rem; font-weight: 700; margin-bottom: 0.4rem; letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--ink), var(--indigo-2), var(--ink));
    background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: shine 6s linear infinite;
    display: flex; align-items: center; gap: 0.7rem;
}
.main-header .bi { -webkit-text-fill-color: var(--indigo); font-size: 2.1rem; }
.page-sub { color: var(--muted); margin: 0 0 1.2rem 0; line-height: 1.5; max-width: 62ch; animation: fadeIn 0.8s ease both; }
.section-title { font-size: 1.15rem; font-weight: 700; color: var(--ink); margin: 1.6rem 0 0.7rem 0;
    display: flex; align-items: center; gap: 0.55rem; animation: fadeUp 0.5s ease both; }
.section-title .ico { width: 32px; height: 32px; border-radius: 9px; background: var(--indigo-soft); color: var(--indigo);
    display: inline-grid; place-items: center; font-size: 1rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #14213d 0%, #1b2c58 100%); }
section[data-testid="stSidebar"] * { color: #e7eaf3; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.25); color: #fff;
    border-radius: 10px; font-weight: 500; transition: all .2s ease; }
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,0.16); border-color: #fff; transform: translateX(3px); }
section[data-testid="stSidebar"] [data-testid="stAlert"] { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); }
.brand { display: flex; align-items: center; gap: 0.6rem; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.8rem; }
.brand-mark { width: 32px; height: 32px; border-radius: 9px; background: #fff; display: grid; place-items: center; color: var(--indigo) !important; }
.brand-mark .bi { color: var(--indigo) !important; }
.side-head { font-weight: 600; margin: 0.4rem 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem; }
.user-block { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0 0.8rem 0; }
.avatar { width: 42px; height: 42px; border-radius: 50%; background: var(--indigo-2); color: #fff !important;
    display: grid; place-items: center; font-weight: 700; animation: avatarRing 2.4s ease-out infinite; }
.pill { display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.75rem; background: rgba(255,255,255,0.15); margin: 0.2rem 0.25rem 0 0; }

/* ── Cards ── */
.card, .incident-card, .recommendation-card, .error-message, .success-message {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.2rem; margin: 0.6rem 0;
    box-shadow: 0 1px 2px rgba(20,33,61,0.05); animation: fadeUp 0.55s ease both; transition: transform .2s ease, box-shadow .2s ease;
}
.card:hover, .incident-card:hover, .recommendation-card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(20,33,61,0.10); }
.incident-card { border-left: 5px solid var(--indigo); }
.recommendation-card { border-left: 5px solid var(--ok); }
.error-message { background: #fff1f0; border-left: 5px solid var(--crit); }
.success-message { background: #effaf4; border-left: 5px solid var(--ok); }
.cause-card { border-left: 5px solid var(--indigo-2); background: linear-gradient(135deg, #fff, #f1f4ff); }
.cause-text { font-size: 1.1rem; font-weight: 600; color: var(--ink); line-height: 1.45; margin: 0.5rem 0 0 0; }
.label { font-size: 0.8rem; color: var(--muted); font-weight: 500; }
.val { color: var(--ink); font-weight: 600; }
ul.factors { margin: 0.4rem 0 0 0; padding-left: 1.2rem; color: #2d3550; line-height: 1.7; }
.inc-grid { display: grid; grid-template-columns: 1fr 150px; gap: 1rem; align-items: center; }
.inc-id { font-weight: 700; color: var(--ink); }
.inc-why { color: #2d3550; font-size: 0.92rem; line-height: 1.5; margin-top: 0.2rem; }
.meter { height: 7px; background: var(--indigo-soft); border-radius: 99px; overflow: hidden; margin-top: 0.35rem; }
.meter > span { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--indigo), var(--indigo-2)); animation: fillBar 1.1s ease both; }
.score { text-align: right; font-size: 0.85rem; color: var(--muted); font-variant-numeric: tabular-nums; }

/* Badges */
.badge { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.18rem 0.65rem; border-radius: 7px; font-size: 0.8rem; font-weight: 600; color: #fff; }
.sev-low { background: var(--ok); } .sev-medium { background: var(--warn); } .sev-high { background: var(--hot); }
.sev-critical { background: var(--crit); animation: pulseRing 1.6s ease-out infinite; }

/* st.metric as cards */
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 0.8rem 1rem;
    box-shadow: 0 1px 2px rgba(20,33,61,0.05); animation: fadeUp 0.55s ease both; transition: transform .2s ease, box-shadow .2s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(20,33,61,0.10); border-color: var(--indigo-2); }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 700; font-size: 1.35rem; }

/* Empty state & example buttons */
.empty { background: rgba(255,255,255,0.85); border: 1px dashed #aab3d6; border-radius: 14px; padding: 1.4rem 1.5rem; margin: 0.5rem 0 1rem 0; animation: fadeUp .6s ease both; }
.empty h3 { margin: 0 0 0.3rem 0; color: var(--ink); font-size: 1.15rem; display: flex; gap: .5rem; align-items: center; }
.empty h3 .bi { color: var(--indigo-2); animation: bob 2.6s ease-in-out infinite; }
.empty p { margin: 0; color: var(--muted); }
.main .stButton button { background: var(--card); border: 1px solid var(--line); color: var(--ink); border-radius: 12px;
    text-align: left; padding: 0.8rem 1rem; font-weight: 500; transition: all .2s ease; }
.main .stButton button:hover { border-color: var(--indigo-2); color: var(--indigo); transform: translateY(-3px); box-shadow: 0 8px 20px rgba(47,63,143,0.14); }

/* Chat */
[data-testid="stChatMessage"] { background: transparent; padding: 0.4rem 0; animation: fadeUp .4s ease both; }
[data-testid="stChatInput"] { border-radius: 14px; }

/* ── Login page ── */
.hero h1 { font-size: 2.4rem; line-height: 1.15; color: var(--ink); margin: 0 0 0.8rem 0; font-weight: 700; letter-spacing: -0.02em; animation: fadeUp .7s ease both; }
.hero p { color: var(--muted); font-size: 1.05rem; line-height: 1.6; max-width: 52ch; animation: fadeUp .7s .1s ease both; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.1rem 0 0.4rem 0; animation: fadeUp .7s .2s ease both; }
.chip { background: rgba(255,255,255,0.8); border: 1px solid var(--line); border-radius: 999px; padding: 0.3rem 0.8rem; font-size: 0.85rem; color: var(--ink); display: inline-flex; gap: .4rem; align-items: center; }
.chip .bi { color: var(--indigo-2); }
.feat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.8rem; margin-top: 1.2rem; }
.feat { background: rgba(255,255,255,0.88); border: 1px solid var(--line); border-radius: 14px; padding: 1rem; backdrop-filter: blur(6px);
    animation: fadeUp .7s ease both; transition: transform .2s ease, box-shadow .2s ease; }
.feat:nth-child(1) { animation-delay: .25s; } .feat:nth-child(2) { animation-delay: .35s; }
.feat:nth-child(3) { animation-delay: .45s; } .feat:nth-child(4) { animation-delay: .55s; }
.feat:hover { transform: translateY(-4px); box-shadow: 0 12px 26px rgba(47,63,143,0.16); }
.feat .fi { width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, var(--indigo), var(--indigo-2)); color: #fff;
    display: grid; place-items: center; font-size: 1.1rem; margin-bottom: 0.6rem; }
.feat b { color: var(--ink); display: block; margin-bottom: 0.2rem; }
.feat span { color: var(--muted); font-size: 0.88rem; line-height: 1.45; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.8rem; margin-top: 0.6rem; }
.step { display: flex; gap: 0.7rem; align-items: flex-start; animation: fadeUp .7s .6s ease both; }
.step .n { flex: none; width: 30px; height: 30px; border-radius: 50%; background: var(--ink); color: #fff; display: grid; place-items: center; font-weight: 700; font-size: .9rem; }
.step b { color: var(--ink); } .step span { display: block; color: var(--muted); font-size: .88rem; line-height: 1.4; }
.login-title { font-size: 1.5rem; font-weight: 700; color: var(--ink); margin: 0 0 0.2rem 0; display: flex; gap: .5rem; align-items: center; animation: fadeUp .6s ease both; }
.login-title .bi { color: var(--indigo-2); }

/* Forms & primary buttons */
[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,0.92); padding: 1.3rem;
    box-shadow: 0 18px 40px rgba(20,33,61,0.12); backdrop-filter: blur(8px); animation: fadeUp .7s .15s ease both; }
.stFormSubmitButton button { background: linear-gradient(135deg, var(--indigo), var(--indigo-2)); border: none; color: #fff;
    border-radius: 10px; font-weight: 600; padding: 0.6rem 1rem; transition: transform .2s ease, box-shadow .2s ease; }
.stFormSubmitButton button:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(47,63,143,0.35); color: #fff; }

@media (max-width: 700px) { .inc-grid { grid-template-columns: 1fr; } .score { text-align: left; } .hero h1 { font-size: 1.8rem; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
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
if 'pending_prompt' not in st.session_state:
    st.session_state.pending_prompt = None

# ── Helpers ──────────────────────────────────────────────────────────────────────

def html(markup: str):
    """Render raw HTML (flattened so Markdown never treats it as a code block)."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)

def e(value, default="N/A") -> str:
    """HTML-escape any value coming from the API."""
    if value is None or value == "":
        return default
    return escape(str(value))

def section(icon: str, text: str):
    html(f'<div class="section-title"><span class="ico"><i class="bi bi-{icon}"></i></span>{text}</div>')

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
    except requests.exceptions.RequestException as e_:
        st.error(f"Connection error: {e_}")
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
    except requests.exceptions.RequestException as e_:
        st.error(f"Connection error: {e_}")
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
    # Animated background layer
    html("""
    <div class="bg-anim">
        <div class="threads"></div>
        <div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
    </div>
    """)

    html('<div class="main-header"><i class="bi bi-activity"></i>Quality AI Diagnostic System</div>')
    st.markdown("---")

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        html("""
        <div class="hero">
            <h1>Find the root cause of a quality problem before the next batch runs.</h1>
            <p>Describe what you see on the floor. Four AI agents read your report, search past incidents and return a diagnosis with ranked next steps.</p>
            <div class="chip-row">
                <span class="chip"><i class="bi bi-shield-lock-fill"></i>Role-based access</span>
                <span class="chip"><i class="bi bi-lightning-charge-fill"></i>Answers in seconds</span>
                <span class="chip"><i class="bi bi-journal-check"></i>Cites past incidents</span>
            </div>
        </div>
        """)

        html("""
        <div class="feat-grid">
            <div class="feat"><div class="fi"><i class="bi bi-chat-left-text-fill"></i></div><b>Plain-language intake</b><span>Loom, batch, metric and symptom are pulled from your own words.</span></div>
            <div class="feat"><div class="fi"><i class="bi bi-search"></i></div><b>Incident matching</b><span>Finds similar historical incidents and scores how close each one is.</span></div>
            <div class="feat"><div class="fi"><i class="bi bi-diagram-3-fill"></i></div><b>Root-cause analysis</b><span>Gives a primary cause, contributing factors and a severity level.</span></div>
            <div class="feat"><div class="fi"><i class="bi bi-list-check"></i></div><b>Prioritised actions</b><span>Recommendations sorted into immediate, short-term and long-term.</span></div>
        </div>
        """)

        section("signpost-split-fill", "How it works")
        html("""
        <div class="steps">
            <div class="step"><div class="n">1</div><div><b>Describe</b><span>Type the problem as you'd tell a colleague.</span></div></div>
            <div class="step"><div class="n">2</div><div><b>Match</b><span>The system searches past incidents.</span></div></div>
            <div class="step"><div class="n">3</div><div><b>Act</b><span>Review the cause and follow the steps.</span></div></div>
        </div>
        """)

    with right:
        html('<div class="login-title"><i class="bi bi-box-arrow-in-right"></i>Login</div>')

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
qa_officer / qa1234
technician / tech123
manager / mgr123
""")

def render_chat_interface():
    """Render main chat interface"""
    # Header
    html('<div class="main-header"><i class="bi bi-activity"></i>Quality AI Diagnostic System</div>')

    # Sidebar
    with st.sidebar:
        html('<div class="brand"><div class="brand-mark"><i class="bi bi-activity"></i></div>Quality AI Diagnostics</div>')

        html('<div class="side-head"><i class="bi bi-person-circle"></i>User Information</div>')
        username = st.session_state.username or "?"
        roles_html = "".join(f'<span class="pill">{e(r)}</span>' for r in st.session_state.user_roles)
        html(f"""
        <div class="user-block">
            <div class="avatar">{e(username[:1].upper())}</div>
            <div><div><b>Username:</b> {e(username)}</div><div><b>Roles:</b> {roles_html}</div></div>
        </div>
        """)

        st.markdown("---")

        html('<div class="side-head"><i class="bi bi-gear-fill"></i>Actions</div>')
        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()

        st.markdown("---")

        html('<div class="side-head"><i class="bi bi-clock-history"></i>Chat History</div>')
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

    # Empty state with example questions (only before the first question)
    if not st.session_state.chat_history:
        html("""
        <div class="empty">
            <h3><i class="bi bi-stars"></i>Start with a question</h3>
            <p>Pick an example or type your own in the box below.</p>
        </div>
        """)
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for i, (col, q) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
            if col.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()

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
    prompt = st.chat_input("Ask a diagnostic question...")
    if st.session_state.pending_prompt:
        prompt, st.session_state.pending_prompt = st.session_state.pending_prompt, None

    if prompt:
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
    diagnosis_info = {}  # defined up front so the Agent 3 fallback never hits a NameError

    # Agent 1: Entity Extraction
    agent1_data = agent_outputs.get("agent1_intake", {})
    if agent1_data and "error" not in agent1_data:
        section("clipboard-data-fill", "Extracted Information")
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
                section("search", "Root Cause Analysis")

                severity_map = {
                    "low": ("check-circle-fill", "sev-low"),
                    "medium": ("exclamation-circle-fill", "sev-medium"),
                    "high": ("exclamation-triangle-fill", "sev-high"),
                    "critical": ("x-octagon-fill", "sev-critical"),
                }
                sev = str(rca.get("severity", "medium")).lower()
                sev_icon, sev_class = severity_map.get(sev, ("question-circle-fill", "sev-medium"))

                factors = "".join(f"<li>{e(f)}</li>" for f in rca.get("contributing_factors", []))
                factors_html = (
                    f'<div class="label" style="margin-top:0.9rem;"><b>Contributing Factors:</b></div><ul class="factors">{factors}</ul>'
                    if factors else ""
                )
                html(f"""
                <div class="card cause-card">
                    <span class="badge {sev_class}"><i class="bi bi-{sev_icon}"></i>Severity: {e(rca.get("severity", "N/A")).upper()}</span>
                    <p class="cause-text"><span class="label">Primary Cause: </span>{e(rca.get("primary_cause"))}</p>
                    {factors_html}
                </div>
                """)

            # Cited Incidents
            cited_incidents = diagnosis_info.get("cited_incidents", [])
            if cited_incidents:
                section("journal-text", "Cited Historical Incidents")
                for incident in cited_incidents:
                    try:
                        score = float(incident.get("similarity_score", 0))
                    except (TypeError, ValueError):
                        score = 0.0
                    pct = max(0, min(100, round(score * 100)))
                    html(f"""
                    <div class="incident-card">
                        <div class="inc-grid">
                            <div>
                                <div class="inc-id"><i class="bi bi-folder2-open"></i> <span class="label">Incident ID:</span> {e(incident.get('incident_id'))}</div>
                                <div class="inc-why"><b>Relevance:</b> {e(incident.get('relevance'))}</div>
                            </div>
                            <div><div class="score">Similarity Score: {score:.3f}</div><div class="meter"><span style="width:{pct}%"></span></div></div>
                        </div>
                    </div>
                    """)

            # Recommendations
            recommendations = diagnosis_info.get("recommendations", [])
            if recommendations:
                section("lightbulb-fill", "Recommendations")
                priority_order = {"immediate": 0, "short-term": 1, "long-term": 2}
                sorted_recommendations = sorted(recommendations, key=lambda x: priority_order.get(x.get("priority", "long-term"), 3))

                priority_map = {
                    "immediate": ("lightning-charge-fill", "sev-critical"),
                    "short-term": ("hourglass-split", "sev-medium"),
                    "long-term": ("calendar-check-fill", "sev-low"),
                }
                for rec in sorted_recommendations:
                    p_icon, p_class = priority_map.get(rec.get("priority", "long-term"), ("circle-fill", "sev-low"))
                    html(f"""
                    <div class="recommendation-card">
                        <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;">
                            <span class="badge {p_class}"><i class="bi bi-{p_icon}"></i>{e(rec.get('priority'))}</span>
                            <span class="val">{e(rec.get('action'))}</span>
                        </div>
                        <div class="inc-why" style="margin-top:0.5rem;"><b>Priority:</b> {e(rec.get('priority'))}<br>
                        <b>Expected Outcome:</b> {e(rec.get('expected_outcome'))}</div>
                    </div>
                    """)

            # Confidence and Notes
            confidence = diagnosis_info.get("confidence", "N/A")
            additional_notes = diagnosis_info.get("additional_notes", "")

            if confidence != "N/A" or additional_notes:
                section("graph-up-arrow", "Additional Information")
                conf_html, notes_html = "", ""
                if confidence != "N/A":
                    conf_pct = {"low": 33, "medium": 66, "high": 100}.get(str(confidence).lower(), 50)
                    conf_html = (
                        f'<div><b>Confidence Level:</b> {e(str(confidence).upper())}</div>'
                        f'<div class="meter"><span style="width:{conf_pct}%"></span></div>'
                    )
                if additional_notes:
                    notes_html = f'<div class="inc-why" style="margin-top:0.7rem;"><b>Notes:</b> {e(additional_notes)}</div>'
                html(f'<div class="card">{conf_html}{notes_html}</div>')

    # Agent 3: Historical Incidents (fallback if Agent 4 doesn't have citations)
    agent3_data = agent_outputs.get("agent3_retrieval", [])
    if agent3_data and isinstance(agent3_data, list) and not diagnosis_info.get("cited_incidents"):
        section("collection-fill", "Similar Historical Incidents")
        for incident in agent3_data[:3]:  # Show top 3
            try:
                score = float(incident.get("similarity_score", 0))
            except (TypeError, ValueError):
                score = 0.0
            pct = max(0, min(100, round(score * 100)))
            html(f"""
            <div class="incident-card">
                <div class="inc-grid">
                    <div>
                        <div class="inc-id"><i class="bi bi-folder2-open"></i> <span class="label">Incident ID:</span> {e(incident.get('incident_id'))}</div>
                        <div class="inc-why"><b>Reason:</b> {e(incident.get('reason'))}</div>
                    </div>
                    <div><div class="score">Similarity Score: {score:.3f}</div><div class="meter"><span style="width:{pct}%"></span></div></div>
                </div>
            </div>
            """)

# ── Main Application ────────────────────────────────────────────────────────────

def main():
    """Main application logic"""
    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()