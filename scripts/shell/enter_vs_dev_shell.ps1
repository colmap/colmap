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
