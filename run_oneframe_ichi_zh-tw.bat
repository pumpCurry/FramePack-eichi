@echo off
echo %~nx0 : 起動開始....

call environment.bat

cd %~dp0webui

"%DIR%\python\python.exe" oneframe_ichi.py --server 127.0.0.1 --lang zh-tw --inbrowser

:done
pause