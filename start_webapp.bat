@echo off
echo ================================================================================
echo     VITALSIGHT WEB DASHBOARD
echo ================================================================================
echo.
echo Starting web server...
echo.

call myenv\Scripts\activate.bat
python webapp.py

pause

