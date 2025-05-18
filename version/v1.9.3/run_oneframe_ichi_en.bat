@echo off

call environment.bat

cd %~dp0webui

"%DIR%\python\python.exe" oneframe_ichi.py --server 127.0.0.1 --lang en --inbrowser

:done
pause