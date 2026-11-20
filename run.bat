@echo off
rem usage:  .\run.bat                      (pick or create a shot in the interface)
rem         .\run.bat "D:\shots\shot01" ["D:\footage\take.mp4"]
rem         set VFXPIPE_VERBOSE=1 first to see full step output in this window
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set V=
if defined VFXPIPE_VERBOSE set V=--verbose
if "%~1"=="" (python -m vfxpipe serve %V%) else if "%~2"=="" (python -m vfxpipe serve %V% %1) else (python -m vfxpipe serve %V% %1 --video %2)
