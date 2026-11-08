@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run "%~dp0app.py"
endlocal
