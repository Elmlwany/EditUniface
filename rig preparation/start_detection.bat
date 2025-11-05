@echo off
echo Starting Rig Preparation Detection System...
echo.

cd /d "%~dp0"
cd ..

echo Activating Python environment...
call .venv\Scripts\activate.bat

echo.
echo Starting detection system with camera index 1...
echo.

python "rig preparation\rig_prep_detection_advanced.py"

echo.
echo Detection system closed.
pause
