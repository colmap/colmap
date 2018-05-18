rem Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
rem All rights reserved.
rem
rem Redistribution and use in source and binary forms, with or without
rem modification, are permitted provided that the following conditions are met:
rem
rem     * Redistributions of source code must retain the above copyright
rem       notice, this list of conditions and the following disclaimer.
rem
rem     * Redistributions in binary form must reproduce the above copyright
rem       notice, this list of conditions and the following disclaimer in the
rem       documentation and/or other materials provided with the distribution.
rem
rem     * Neither the name of the ETH Zurich and UNC Chapel Hill nor the
rem       names of its contributors may be used to endorse or promote products
rem       derived from this software without specific prior written permission.
rem
rem THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
rem AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
rem IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
rem ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
rem DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
rem (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
rem LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
rem ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
rem (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
rem THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
rem
rem Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

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
