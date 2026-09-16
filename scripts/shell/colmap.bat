@echo off
rem SPDX-License-Identifier: BSD-3-Clause

set SCRIPT_PATH=%~dp0

set PATH=%SCRIPT_PATH%\bin;%PATH%
set QT_PLUGIN_PATH=%SCRIPT_PATH%\plugins;%QT_PLUGIN_PATH%

set ARGUMENTS=%*
if "%ARGUMENTS%"=="" set ARGUMENTS=gui

"%SCRIPT_PATH%\bin\colmap" %ARGUMENTS%
