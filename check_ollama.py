"""
Simple diagnostic script to check Ollama connectivity
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

print("=" * 60)
print("OLLAMA CONNECTION DIAGNOSTIC")
print("=" * 60)
print(f"OLLAMA_URL: {OLLAMA_URL}")
print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
print()

# Check if Ollama is running
print("1. Checking if Ollama service is running...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("[OK] Ollama service is running")
        models = response.json().get("models", [])
        print(f"Available models: {[m['name'] for m in models]}")
    else:
        print(f"[FAIL] Ollama service returned status {response.status_code}")
except Exception as e:
    print(f"[FAIL] Cannot connect to Ollama: {e}")
    print("Make sure Ollama is installed and running:")
    print("  - Download from https://ollama.com")
    print("  - Run: ollama serve")
    exit(1)

print()

# Check if the configured model is available
print("2. Checking if configured model is available...")
try:
    models = response.json().get("models", [])
    model_names = [m['name'] for m in models]
    if OLLAMA_MODEL in model_names or f"{OLLAMA_MODEL}:latest" in model_names:
        print(f"[OK] Model '{OLLAMA_MODEL}' is available")
    else:
        print(f"[FAIL] Model '{OLLAMA_MODEL}' not found in available models")
        print(f"Available models: {model_names}")
        print(f"Consider changing OLLAMA_MODEL in .env to one of: {model_names}")
except Exception as e:
    print(f"[FAIL] Error checking model: {e}")

print()

# Test a simple generation
print("3. Testing simple text generation...")
try:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Say 'Hello' in one word.",
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        print(f"[OK] Generation successful")
        print(f"Response: {result.get('response', 'N/A')}")
    else:
        print(f"[FAIL] Generation failed with status {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"[FAIL] Generation error: {e}")

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
