# Activate virtual environment
& "C:\KINDRED-LLM\venv\Scripts\Activate.ps1"

# Navigate to repo directory (adjust if not cloned here)
Set-Location -Path "C:\Path\To\KINDRED-Optimization-Suite"  # Replace with your actual path

# Pull latest from GitHub
git pull origin main

# Update dependencies (using requirements.txt if present, or manual pip)
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt --upgrade
} else {
    pip install --upgrade torch transformers optuna einops deap datasets yfinance pytesseract SpeechRecognition streamlit xgboost pandas numpy matplotlib seaborn
}

# Verify GPU (outputs torch.cuda.is_available())
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# Restart Streamlit dashboard if running
Stop-Process -Name "python" -Force  # Caution: Stops all Python processes; refine if needed
streamlit run kindred_dashboard.py --server.port 8501

Write-Output "Environment updated successfully."