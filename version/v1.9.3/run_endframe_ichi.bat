@echo off

call environment.bat

cd %~dp0webui

"%DIR%\python\python.exe" endframe_ichi.py --server 127.0.0.1 --inbrowser

:done
pause