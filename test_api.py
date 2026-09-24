"""
Test script for the Quality AI Diagnostic API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_login():
    """Test login and get token"""
    print("Testing login endpoint...")
    
    # Test with qa_officer
    response = requests.post(f"{BASE_URL}/token", json={
        "username": "qa_officer",
        "password": "qa123"
    })
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        token_data = response.json()
        print(f"Response: {json.dumps(token_data, indent=2)}")
        return token_data["access_token"]
    else:
        print(f"Error: {response.text}")
        return None

def test_diagnose(token):
    """Test diagnose endpoint"""
    print("Testing diagnose endpoint...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{BASE_URL}/diagnose", 
                           headers=headers,
                           json={
                               "query": "Why is Loom 12 producing inconsistent elongation in Batch B102?"
                           })
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_role_protected_endpoints():
    """Test role-based access control"""
    print("Testing role-based access control...")
    
    # Test admin endpoint with different users
    users = [
        ("qa_officer", "qa123"),
        ("technician", "tech123"),
        ("manager", "mgr123")
    ]
    
    for username, password in users:
        print(f"\nTesting with user: {username}")
        
        # Login
        token_response = requests.post(f"{BASE_URL}/token", json={
            "username": username,
            "password": password
        })
        
        if token_response.status_code != 200:
            print(f"  Login failed: {token_response.text}")
            continue
            
        token = token_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test admin endpoint
        admin_response = requests.get(f"{BASE_URL}/admin/stats", headers=headers)
        print(f"  Admin stats: {admin_response.status_code} - {'Allowed' if admin_response.status_code == 200 else 'Denied'}")
        
        # Test equipment endpoint
        tech_response = requests.get(f"{BASE_URL}/tech/equipment", headers=headers)
        print(f"  Equipment info: {tech_response.status_code} - {'Allowed' if tech_response.status_code == 200 else 'Denied'}")

def test_input_sanitization():
    """Test input sanitization"""
    print("Testing input sanitization...")
    
    # Get token first
    token_response = requests.post(f"{BASE_URL}/token", json={
        "username": "qa_officer",
        "password": "qa123"
    })
    
    if token_response.status_code != 200:
        print("Failed to get token")
        return
        
    token = token_response.json()["access_token"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test malicious inputs
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "query with < > \" ' characters"
    ]
    
    for malicious_input in malicious_inputs:
        print(f"\nTesting input: {malicious_input}")
        response = requests.post(f"{BASE_URL}/diagnose",
                               headers=headers,
                               json={"query": malicious_input})
        print(f"  Status: {response.status_code}")
        if response.status_code == 422:
            print(f"  Blocked: {response.json()['detail'][0]['msg']}")

if __name__ == "__main__":
    print("=" * 60)
    print("QUALITY AI DIAGNOSTIC API TEST")
    print("=" * 60)
    print()
    
    try:
        # Test basic endpoints
        test_health()
        
        # Test authentication
        token = test_login()
        
        if token:
            # Test main functionality
            test_diagnose(token)
            
            # Test role-based access
            test_role_protected_endpoints()
            
            # Test input sanitization
            test_input_sanitization()
        else:
            print("Failed to authenticate, skipping authenticated tests")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
