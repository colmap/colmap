$CURRDIR = $PWD

$COMPILER_TOOLS_DIR = "${env:COMPILER_CACHE_DIR}/bin"
New-Item -ItemType Directory -Force -Path ${COMPILER_TOOLS_DIR}
$env:Path = "${COMPILER_TOOLS_DIR};" + $env:Path

If (!(Test-Path -path "${COMPILER_TOOLS_DIR}/ccache.exe" -PathType Leaf)) {
    .github/workflows/install-ccache.ps1 -Destination "${COMPILER_TOOLS_DIR}"
}

# Setup vcpkg
cd ${CURRDIR}
git clone https://github.com/microsoft/vcpkg ${env:VCPKG_INSTALLATION_ROOT}
cd ${env:VCPKG_INSTALLATION_ROOT}
git checkout "${env:VCPKG_COMMIT_ID}"
./bootstrap-vcpkg.bat

cd ${CURRDIR}
& "./scripts/shell/enter_vs_dev_shell.ps1"
& "${env:VCPKG_INSTALLATION_ROOT}/vcpkg.exe" integrate install

# Build COLMAP
mkdir build
cd build
cmake .. `
    -GNinja `
    -DCMAKE_MAKE_PROGRAM=ninja `
    -DCUDA_ENABLED="OFF" `
    -DGUI_ENABLED="OFF" `
    -DCGAL_ENABLED="OFF" `
    -DLSD_ENABLED="OFF" `
    -DCMAKE_BUILD_TYPE="Release" `
    -DCMAKE_TOOLCHAIN_FILE="${env:CMAKE_TOOLCHAIN_FILE}" `
    -DVCPKG_TARGET_TRIPLET="${env:VCPKG_TARGET_TRIPLET}"
ninja install

ccache --show-stats --verbose
ccache --evict-older-than 1d
ccache --show-stats --verbose
