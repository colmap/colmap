set SCRIPT_PATH=%~dp0
set PATH=%SCRIPT_PATH%\lib;%PATH%
set QT_PLUGIN_PATH=%SCRIPT_PATH%\lib;%QT_PLUGIN_PATH%

for %%i in (%SCRIPT_PATH%\bin\*_test.exe) do (
    %%i

    if %errorlevel% neq 0 goto end
)

:end
pause
