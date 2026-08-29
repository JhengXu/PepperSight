@echo off
setlocal
set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%backend\.venv\Scripts\python.exe" (
  echo [ERROR] Backend environment is missing. Run setup.bat first.
  pause
  exit /b 1
)

if not exist "%PROJECT_DIR%frontend\node_modules" (
  echo [ERROR] Frontend dependencies are missing. Run setup.bat first.
  pause
  exit /b 1
)

start "Qianjiao Backend" cmd /k "cd /d ""%PROJECT_DIR%backend"" && .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"
start "Qianjiao Frontend" cmd /k "cd /d ""%PROJECT_DIR%frontend"" && npm run dev"

echo.
echo Qianjiao inspection services are starting...
echo Dashboard: http://localhost:3000/inspection
echo API docs:  http://localhost:8000/docs
echo.
timeout /t 4 /nobreak >nul
start "" "http://localhost:3000/inspection"
endlocal
