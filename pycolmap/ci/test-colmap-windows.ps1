& "./scripts/shell/enter_vs_dev_shell.ps1"
& "${env:VCPKG_INSTALLATION_ROOT}/vcpkg.exe" integrate install

& python -c "import pycolmap; print(pycolmap.__version__)"
