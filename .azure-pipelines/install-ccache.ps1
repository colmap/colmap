[CmdletBinding()]
param (
    [Parameter(Mandatory = $true)]
    [string] $Destination
)

$version = "4.8"
$folder = "ccache-$version-windows-x86_64"
$url = "https://github.com/ccache/ccache/releases/download/v$version/$folder.zip"
$expectedSha256 = "A2B3BAB4BB8318FFC5B3E4074DC25636258BC7E4B51261F7D9BEF8127FDA8309"

$ErrorActionPreference = "Stop"

try {
    New-Item -Path "$Destination" -ItemType Container -ErrorAction SilentlyContinue

    Write-Host "Download CCache"
    $zipFilePath = Join-Path "$env:TEMP" "$folder.zip"
    Invoke-WebRequest -Uri $url -UseBasicParsing -OutFile "$zipFilePath" -MaximumRetryCount 3

    $hash = Get-FileHash $zipFilePath -Algorithm "sha256"
    if ($hash.Hash -ne $expectedSha256) {
        throw "File $Path hash $hash.Hash did not match expected hash $expectedHash"
    }

    Write-Host "Unzip CCache"
    Expand-Archive -Path "$zipFilePath" -DestinationPath "$env:TEMP"

    Write-Host "Move CCache"
    Move-Item -Force "$env:TEMP/$folder/ccache.exe" "$Destination"
    Remove-Item "$zipFilePath"
    Remove-Item -Recurse "$env:TEMP/$folder"
}
catch {
    Write-Host "Installation failed with an error"
    $_.Exception | Format-List
    exit -1
}
