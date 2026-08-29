@echo off
setlocal
set "PROJECT_DIR=%~dp0"

echo [1/2] Preparing Python backend...
cd /d "%PROJECT_DIR%backend"
if not exist .venv py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [2/2] Installing frontend packages...
cd /d "%PROJECT_DIR%frontend"
call npm install
if errorlevel 1 goto :failed

echo.
echo Setup complete. Double-click start-demo.bat to launch the system.
pause
exit /b 0

:failed
echo.
echo Setup failed. Check the network output above and retry.
pause
exit /b 1

