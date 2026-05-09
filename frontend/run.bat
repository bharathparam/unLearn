@echo off
echo.
echo  ==========================================
echo   NeuralLifecycle Framework v4.2.1
echo   Starting AI Research Dashboard...
echo  ==========================================
echo.

echo Starting Streamlit Frontend on port 8502...
python -m streamlit run app.py --server.port 8502
pause
