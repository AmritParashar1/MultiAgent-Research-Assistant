@echo off
:: ============================================================
:: Multi-Agent Research Assistant — App Launcher
:: Run this file to start the Streamlit app using the project venv.
:: ============================================================
echo Starting Research Assistant...
call "%~dp0.venv\Scripts\activate.bat"
python -m streamlit run "%~dp0ui\streamlit_app.py"
