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
    # CI only: the Visual Studio toolchain bundles an LLVM flang Fortran compiler
    # that miscompiles LAPACK's *gedmd routines (e.g. "'ssum' is not an object
    # that can appear in an expression"). vcpkg builds each port in its own
    # environment (regenerated via vcvars), which puts this flang on the PATH, so
    # vcpkg_find_fortran detects it instead of falling back to a working MinGW
    # gfortran and the lapack-reference build fails. Remove the bundled flang in CI
    # so the MinGW gfortran fallback kicks in. COLMAP itself builds with MSVC
    # (cl.exe) and does not use flang. We gate on CI so local Visual Studio
    # installations are left untouched.
    # See https://github.com/llvm/llvm-project/issues/201254 and
    # https://developercommunity.microsoft.com/t/11105096.
    if ($env:GITHUB_ACTIONS -eq "true") {
        $llvmDir = Join-Path $vsInstance.installationPath "VC/Tools/Llvm"
        if (Test-Path $llvmDir) {
            Get-ChildItem -Path $llvmDir -Recurse -Filter "flang*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
                try {
                    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop
                    Write-Host "Removed bundled flang: $($_.FullName)"
                } catch {
                    Write-Warning "Failed to remove bundled flang $($_.FullName): $_"
                }
            }
        }
    }

    Set-Location $prevCwd

    $env:VisualStudioDevShell = $true
}
