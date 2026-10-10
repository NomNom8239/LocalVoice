@echo off
setlocal
title Irodori-TTS VoiceDesign

set "APP_DIR=%~dp0Irodori-TTS"

if not exist "%APP_DIR%\.venv\Scripts\python.exe" (
    echo [ERROR] Python environment not found.
    pause
    exit /b 1
)

if not exist "%APP_DIR%\gradio_app_voicedesign.py" (
    echo [ERROR] VoiceDesign script not found.
    pause
    exit /b 1
)

cd /d "%APP_DIR%"

echo Starting Irodori-TTS VoiceDesign...
echo URL: http://127.0.0.1:7861
echo.

".venv\Scripts\python.exe" "gradio_app_voicedesign.py" --server-name 127.0.0.1 --server-port 7861

echo.
echo Irodori-TTS stopped.
pause