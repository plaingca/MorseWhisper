# Install PyTorch with CUDA support for MorseWhisper
# For RTX 4080 with CUDA 12.9

Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Green

# First upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.9)
python -m pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the requirements
Write-Host "Installing other requirements..." -ForegroundColor Green
python -m pip install -r requirements.txt

# Verify CUDA is available
Write-Host "`nVerifying CUDA installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 