if (!$env:VisualStudioDevShell) {
    $vswhere = "${Env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
    if (!(Test-Path $vswhere)) {
        throw "Failed to find vswhere.exe"
    }

    & $vswhere -latest -format json
    $vsInstance = & $vswhere -latest -format json | ConvertFrom-Json 
    if ($LASTEXITCODE) {
        throw "vswhere.exe returned exit code $LASTEXITCODE"
    }

    Import-Module "$($vsInstance.installationPath)/Common7/Tools/Microsoft.VisualStudio.DevShell.dll"
    $prevCwd = Get-Location
    try {
        Enter-VsDevShell $vsInstance.instanceId -DevCmdArguments "-no_logo -host_arch=amd64 -arch=amd64"
    } catch {
        Write-Host $_
        Write-Error "Failed to enter Visual Studio Dev Shell"
        exit 1
    }
    Set-Location $prevCwd

    $env:VisualStudioDevShell = $true
}

# The Visual Studio toolchain bundles an LLVM flang Fortran compiler that
# miscompiles LAPACK's *gedmd routines (e.g. "'ssum' is not an object that can
# appear in an expression"). When flang is on the PATH, vcpkg's
# vcpkg_find_fortran picks it up instead of falling back to a working MinGW
# gfortran, breaking the lapack-reference build. Remove the bundled LLVM tools
# from the PATH so vcpkg uses MinGW gfortran. COLMAP itself builds with MSVC
# (cl.exe) and does not need these tools.
# See https://github.com/llvm/llvm-project/issues/201254 and
# https://developercommunity.microsoft.com/t/11105096.
$env:Path = ($env:Path -split ';' | Where-Object { $_ -notmatch '\\VC\\Tools\\Llvm\\' }) -join ';'
