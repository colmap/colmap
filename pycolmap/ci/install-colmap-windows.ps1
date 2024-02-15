$CURRDIR = $PWD

$COMPILER_TOOLS_DIR = "${env:COMPILER_CACHE_DIR}/bin"
New-Item -ItemType Directory -Force -Path ${COMPILER_TOOLS_DIR}
$env:Path = "${COMPILER_TOOLS_DIR};" + $env:Path

$NINJA_PATH = "${COMPILER_TOOLS_DIR}/ninja.exe"
If (!(Test-Path -path ${NINJA_PATH} -PathType Leaf)) {
    $zip_path = "${env:TEMP}/ninja.zip"
    $url = "https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-win.zip"
    curl.exe -L -o ${zip_path} ${url}
    Expand-Archive -LiteralPath ${zip_path} -DestinationPath ${COMPILER_TOOLS_DIR}
    Remove-Item ${zip_path}
}
If (!(Test-Path -path "${COMPILER_TOOLS_DIR}/ccache.exe" -PathType Leaf)) {
    # For some reason this CI runs an earlier PowerShell version that is
    # not compatible with colmap/.github/workflows/install-ccache.ps1
    $folder = "ccache-4.8-windows-x86_64"
    $url = "https://github.com/ccache/ccache/releases/download/v4.8/${folder}.zip"
    $zip_path = "${env:TEMP}/${folder}.zip"
    $folder_path = "${env:TEMP}/${folder}"
    curl.exe -L -o ${zip_path} ${url}
    Expand-Archive -LiteralPath ${zip_path} -DestinationPath "$env:TEMP"
    Move-Item -Force "${folder_path}/ccache.exe" ${COMPILER_TOOLS_DIR}
    Remove-Item ${zip_path}
    Remove-Item -Recurse ${folder_path}
}

# Build the dependencies
cd ${CURRDIR}
git clone https://github.com/microsoft/vcpkg ${env:VCPKG_INSTALLATION_ROOT}
cd ${env:VCPKG_INSTALLATION_ROOT}
git checkout "${env:VCPKG_COMMIT_ID}"
./bootstrap-vcpkg.bat

cd ${CURRDIR}
& "./scripts/shell/enter_vs_dev_shell.ps1"

[System.Collections.ArrayList]$DEPS = Get-Content -Path "./pycolmap/ci/vcpkg-dependencies.txt"
& "${env:VCPKG_INSTALLATION_ROOT}/vcpkg.exe" install --recurse --clean-after-build `
    --triplet="${env:VCPKG_TARGET_TRIPLET}" @DEPS
& "${env:VCPKG_INSTALLATION_ROOT}/vcpkg.exe" integrate install

# Build COLMAP
mkdir build
cd build
cmake .. `
    -GNinja `
    -DCMAKE_MAKE_PROGRAM="${NINJA_PATH}" `
    -DCMAKE_BUILD_TYPE="Release" `
    -DCUDA_ENABLED="OFF" `
    -DCGAL_ENABLED="OFF" `
    -DGUI_ENABLED="OFF" `
    -DCMAKE_TOOLCHAIN_FILE="${env:CMAKE_TOOLCHAIN_FILE}" `
    -DVCPKG_TARGET_TRIPLET="${env:VCPKG_TARGET_TRIPLET}"
& ${NINJA_PATH} install

ccache --show-stats --verbose
ccache --evict-older-than 1d
ccache --show-stats --verbose
