@echo off
setlocal
title FlyDecoder first-time setup
cd /d "%~dp0"
set "FLYBRAIN_DEVICE=cpu"
set "FLYBRAIN_DATA=%~dp0data\malecns"
set "PYTHONIOENCODING=utf-8"

set "UV="
for /f "delims=" %%i in ('where uv.exe 2^>nul') do if not defined UV set "UV=%%i"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV=%USERPROFILE%\.cargo\bin\uv.exe"
if defined UV goto setup

echo Installing uv, the Python environment manager. Internet is needed for first-time setup.
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072; irm https://astral.sh/uv/install.ps1 | iex"
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV echo Could not install uv. See the messages above.
if not defined UV goto failed

:setup
if not exist ".venv\Scripts\python.exe" (
  echo Installing Python 3.12 into the local project environment...
  "%UV%" venv --python 3.12 .venv
  if errorlevel 1 goto failed
)

echo Installing FlyDecoder CPU runtime dependencies...
"%UV%" pip install --python ".venv\Scripts\python.exe" -e .
if errorlevel 1 goto failed

echo Downloading the three required MaleCNS tables and checking SHA-256...
".venv\Scripts\python.exe" -m flydecoder_app.download_data
if errorlevel 1 goto failed

echo Building the local MaleCNS connectome cache. This is a one-time step.
".venv\Scripts\python.exe" -m flybrain.connectome
if errorlevel 1 goto failed

echo Setup is complete. Starting FlyDecoder.
call run.bat
goto end

:failed
echo.
echo Setup did not complete. Rerun this file to resume; downloaded files are kept.
pause
exit /b 1

:end
endlocal
