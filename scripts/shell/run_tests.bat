set PATH=lib;%PATH%
set QT_PLUGIN_PATH=lib;%QT_PLUGIN_PATH%
for %%i in (bin/*_test.exe) do bin\%%i
pause
