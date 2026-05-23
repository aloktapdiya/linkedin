# Graph RAG POC - Windows Quick Start
# Run this from the graph_rag_poc folder:
#   powershell -ExecutionPolicy Bypass -File run.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Graph RAG POC - Local Setup & Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---- 1. Check Python ----
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pyver = python --version 2>&1
    Write-Host "      OK: $pyver" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Python not found. Install from https://python.org" -ForegroundColor Red
    exit 1
}

# ---- 2. Check Ollama ----
Write-Host "[2/5] Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaVer = ollama --version 2>&1
    Write-Host "      OK: $ollamaVer" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Ollama not found. Install from https://ollama.com" -ForegroundColor Red
    exit 1
}

# ---- 3. Ensure llama3.2:1b is pulled ----
Write-Host "[3/5] Checking llama3.2:1b model..." -ForegroundColor Yellow
$models = ollama list 2>&1
if ($models -match "llama3.2:1b") {
    Write-Host "      OK: llama3.2:1b already present" -ForegroundColor Green
} else {
    Write-Host "      Pulling llama3.2:1b (~1.3 GB)..." -ForegroundColor Yellow
    ollama pull llama3.2:1b
    Write-Host "      OK: llama3.2:1b pulled" -ForegroundColor Green
}

# ---- 4. Start Ollama serve (if not already running) ----
Write-Host "[4/5] Starting Ollama server..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "      OK: Ollama already running on port 11434" -ForegroundColor Green
} catch {
    Write-Host "      Launching ollama serve in background..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "      OK: Ollama serve started" -ForegroundColor Green
}

# ---- 5. Install Python dependencies ----
Write-Host "[5/5] Installing Python dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERROR: pip install failed" -ForegroundColor Red
    exit 1
}
Write-Host "      OK: Dependencies installed" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Setup complete! Choose what to run:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [A] Launch Streamlit web app  (recommended)" -ForegroundColor Green
Write-Host "  [B] Run full benchmark (20 questions)" -ForegroundColor White
Write-Host "  [Q] Ask a single question" -ForegroundColor White
Write-Host "  [T] Run tests" -ForegroundColor White
Write-Host "  [X] Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice"

switch ($choice.ToUpper()) {
    "A" {
        Write-Host ""
        Write-Host "Launching Streamlit app at http://localhost:8501 ..." -ForegroundColor Cyan
        streamlit run app.py
    }
    "B" {
        Write-Host ""
        Write-Host "Running benchmark..." -ForegroundColor Cyan
        python main.py --mode benchmark
    }
    "Q" {
        Write-Host ""
        $question = Read-Host "Enter your question"
        python main.py --mode query --question $question
    }
    "T" {
        Write-Host ""
        Write-Host "Running tests..." -ForegroundColor Cyan
        pytest tests/ -v
    }
    "X" {
        Write-Host "Bye!" -ForegroundColor Gray
    }
    default {
        Write-Host "Launching Streamlit app..." -ForegroundColor Cyan
        streamlit run app.py
    }
}
