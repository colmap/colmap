set SCRIPT_PATH=%~dp0
set PATH=%SCRIPT_PATH%\lib;%PATH%
set QT_PLUGIN_PATH=%SCRIPT_PATH%\lib;%QT_PLUGIN_PATH%
for %%i in (%SCRIPT_PATH%\*_test.exe) do bin\%%i
pause
