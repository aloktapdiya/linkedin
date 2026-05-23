#!/bin/bash
# Graph RAG POC - Linux/Mac Quick Start
# Usage: bash run.sh

set -e

echo ""
echo "========================================"
echo "   Graph RAG POC - Local Setup & Run"
echo "========================================"
echo ""

# ---- 1. Check Python ----
echo "[1/5] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "      ERROR: python3 not found"
    exit 1
fi
echo "      OK: $(python3 --version)"

# ---- 2. Check Ollama ----
echo "[2/5] Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "      ERROR: ollama not found. Install from https://ollama.com"
    exit 1
fi
echo "      OK: $(ollama --version)"

# ---- 3. Ensure llama3.2:1b is pulled ----
echo "[3/5] Checking llama3.2:1b model..."
if ollama list | grep -q "llama3.2:1b"; then
    echo "      OK: llama3.2:1b already present"
else
    echo "      Pulling llama3.2:1b (~1.3 GB)..."
    ollama pull llama3.2:1b
fi

# ---- 4. Start Ollama serve (if not running) ----
echo "[4/5] Starting Ollama server..."
if curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
    echo "      OK: Ollama already running on port 11434"
else
    echo "      Launching ollama serve in background..."
    ollama serve &>/dev/null &
    sleep 3
    echo "      OK: Ollama serve started"
fi

# ---- 5. Install Python dependencies ----
echo "[5/5] Installing Python dependencies..."
pip3 install -q -r requirements.txt
echo "      OK: Dependencies installed"

echo ""
echo "========================================"
echo "   Setup complete! Choose what to run:"
echo "========================================"
echo ""
echo "  [A] Launch Streamlit web app  (recommended)"
echo "  [B] Run full benchmark (20 questions)"
echo "  [Q] Ask a single question"
echo "  [T] Run tests"
echo "  [X] Exit"
echo ""
read -p "Enter choice: " choice

case "${choice^^}" in
    A)
        echo ""
        echo "Launching Streamlit app at http://localhost:8501 ..."
        streamlit run app.py
        ;;
    B)
        echo ""
        echo "Running benchmark..."
        python3 main.py --mode benchmark
        ;;
    Q)
        echo ""
        read -p "Enter your question: " question
        python3 main.py --mode query --question "$question"
        ;;
    T)
        echo ""
        echo "Running tests..."
        pytest tests/ -v
        ;;
    *)
        echo "Launching Streamlit app..."
        streamlit run app.py
        ;;
esac
