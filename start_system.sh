#!/bin/bash

# Startup script for Quality AI Diagnostic System
# This script starts both the FastAPI backend and Streamlit frontend

echo "Starting Quality AI Diagnostic System..."
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create it first."
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Start FastAPI backend in background
echo "Starting FastAPI backend..."
python api/main.py &
API_PID=$!
echo "FastAPI backend started with PID: $API_PID"

# Wait a moment for the API to start
sleep 3

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "FastAPI backend is healthy"
else
    echo "Warning: FastAPI backend may not be running properly"
fi

# Start Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run ui/app.py &
UI_PID=$!
echo "Streamlit frontend started with PID: $UI_PID"

echo "=========================================="
echo "System started successfully!"
echo "API: http://localhost:8000"
echo "UI: http://localhost:8501"
echo "=========================================="
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo "Stopping services..."
    kill $API_PID 2>/dev/null
    kill $UI_PID 2>/dev/null
    echo "Services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for both processes
wait