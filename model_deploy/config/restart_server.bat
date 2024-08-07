@echo off

REM Stop the running Django server
echo Stopping Django server...
taskkill /IM python.exe /F

REM Wait for a moment to ensure the process is terminated
timeout /t 2

REM Restart the Django server
echo Restarting Django server...
python manage.py runserver