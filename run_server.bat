@echo off
chcp 65001 >nul
:: ============================================================
:: run_server.bat -- Start ROME API Server
:: Usage: .\run_server.bat  (from PowerShell)
::        run_server.bat    (from CMD)
:: ============================================================

echo.
echo ============================================================
echo   ROME Model Editing API Server
echo ============================================================
echo.

:: -- Step 1: Activate virtual environment -------------------
echo [1/3] Activating virtual environment...
call "%~dp0rome_env\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate rome_env. Run setup_env.bat first.
    exit /b 1
)
echo       OK -- rome_env activated.
echo.

:: -- Step 2: Check dependencies -----------------------------
echo [2/3] Checking dependencies...
python -c "import fastapi, uvicorn, pydantic" 2>NUL
if errorlevel 1 (
    echo       Installing missing packages...
    pip install fastapi "uvicorn[standard]" "pydantic>=2.0.0" httpx --quiet
)
echo       OK -- all packages present.
echo.

:: -- Step 3: Check port 8000 is free ------------------------
echo [3/3] Checking port 8000...
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo WARNING: Port 8000 is already in use!
    echo Run this in PowerShell to free it:
    echo   Get-NetTCPConnection -LocalPort 8000 ^| Select OwningProcess ^| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
    echo.
    echo Or change the port below and retry.
    exit /b 1
)
echo       OK -- port 8000 is free.
echo.

:: -- Start the server ---------------------------------------
echo ============================================================
echo   Local URL  : http://localhost:8000
echo   Swagger UI : http://localhost:8000/docs
echo   ReDoc      : http://localhost:8000/redoc
echo.
echo   TO EXPOSE WITH NGROK (run in a separate terminal):
echo     ngrok http 8000
echo.
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.

python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
