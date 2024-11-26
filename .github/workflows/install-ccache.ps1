[CmdletBinding()]
param (
    [Parameter(Mandatory = $true)]
    [string] $Destination
)

$version = "4.10.2"
$folder="ccache-$version-windows-x86_64"
$url = "https://github.com/ccache/ccache/releases/download/v$version/$folder.zip"
$expectedSha256 = "6252F081876A9A9F700FAE13A5AEC5D0D486B28261D7F1F72AC11C7AD9DF4DA9"

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
