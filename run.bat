@echo off
setlocal
title FlyDecoder
cd /d "%~dp0"
set "FLYBRAIN_DEVICE=cpu"
set "FLYBRAIN_DATA=%~dp0data\malecns"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
  echo First run: preparing Python, FlyBrain, and MaleCNS data.
  call install_flydecoder.bat
  if errorlevel 1 exit /b 1
  exit /b 0
)

if not exist "%FLYBRAIN_DATA%\body-annotations-male-cns-v1.0-minconf-0.5.feather" goto fetch_data
if not exist "%FLYBRAIN_DATA%\body-neurotransmitters-male-cns-v1.0.feather" goto fetch_data
if not exist "%FLYBRAIN_DATA%\connectome-weights-male-cns-v1.0-minconf-0.5.feather" goto fetch_data
goto data_ready
:fetch_data
".venv\Scripts\python.exe" -m flydecoder_app.download_data
if errorlevel 1 goto failed
:data_ready
if not exist "cache\W_post_pre.npz" (
  echo Building the connectome cache once...
  ".venv\Scripts\python.exe" -m flybrain.connectome
  if errorlevel 1 goto failed
)

".venv\Scripts\python.exe" -m flydecoder_app.app
goto end

:failed
echo.
echo FlyDecoder could not prepare MaleCNS. Rerun install_flydecoder.bat.
pause
exit /b 1

:end
endlocal
