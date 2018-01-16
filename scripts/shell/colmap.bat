@echo off

set SCRIPT_PATH=%~dp0

set PATH=%SCRIPT_PATH%\lib;%PATH%
set QT_PLUGIN_PATH=%SCRIPT_PATH%\lib;%QT_PLUGIN_PATH%

set COMMAND=%1
set ARGUMENTS=
shift
:extract_argument_loop
if "%1"=="" goto after_extract_argument_loop
set ARGUMENTS=%ARGUMENTS% %1
shift
goto extract_argument_loop
:after_extract_argument_loop

if "%COMMAND%"=="" set COMMAND=gui

"%SCRIPT_PATH%\bin\colmap" %COMMAND% %ARGUMENTS%
