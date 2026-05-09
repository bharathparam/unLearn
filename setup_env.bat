@echo off
echo Setting up ROME virtual environment for Qwen2.5-1.5B-Instruct...
echo.

REM Create virtual environment
python -m venv rome_env

REM Activate virtual environment
call rome_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support (for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other requirements
pip install -r requirements.txt

echo.
echo Setup complete! Activate the environment with: rome_env\Scripts\activate.bat
