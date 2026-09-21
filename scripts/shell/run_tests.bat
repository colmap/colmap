@echo off
rem SPDX-License-Identifier: BSD-3-Clause

set SCRIPT_PATH=%~dp0

set PATH=%SCRIPT_PATH%\bin;%PATH%
set QT_PLUGIN_PATH=%SCRIPT_PATH%\plugins;%QT_PLUGIN_PATH%

@echo on

for %%i in (%SCRIPT_PATH%\bin\*_test.exe) do (
    %%i

    if %errorlevel% neq 0 goto end
)

:end
pause
