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

if "%COMMAND%"=="" set COMMAND=colmap.exe
if "%COMMAND%"=="gui" set COMMAND=colmap.exe
if exist "%SCRIPT_PATH%\bin\%COMMAND%.exe" set COMMAND=%COMMAND%.exe
if "%COMMAND%"=="help" goto show_help
if "%COMMAND%"=="-h" goto show_help
if "%COMMAND%"=="--help" goto show_help
if not exist "%SCRIPT_PATH%\bin\%COMMAND%" goto show_help 

@echo on

"%SCRIPT_PATH%\bin\%COMMAND%" %ARGUMENTS%

@echo off
goto :eof

:show_help
    echo COLMAP -- Structure-from-Motion and Multi-View Stereo
    echo[
    echo Usage:
    echo   colmap.bat [command] [options]
    echo[
    echo Example usage:
    echo   colmap.bat help [ -h, --help ]
    echo   colmap.bat gui
    echo   colmap.bat gui -h [ --help ]
    echo   colmap.bat feature_extractor --image_path IMAGES --database_path DATABASE
    echo   colmap.bat exhaustive_matcher --database_path DATABASE
    echo   colmap.bat mapper --image_path IMAGES --database_path DATABASE --export_path EXPORT
    echo   ...
    echo[
    echo Documentation:
    echo   https://colmap.github.io/
    echo[
    echo Available commands:
    setlocal enabledelayedexpansion
    for /r %%i in (%SCRIPT_PATH%\bin\*.exe) do (
        set filename=%%i
        set filename_without_exe=!filename:.exe=!
        set filename_without_test=!filename:_test=!
        if not !filename_without_exe!==!filename! (
            if !filename_without_test!==!filename! (
                echo   %%~ni
            )
        )
    )
    endlocal
    goto :eof
