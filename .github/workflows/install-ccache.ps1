[CmdletBinding()]
param (
    [Parameter(Mandatory = $true)]
    [string] $Destination
)

$version = "4.13.6"
$folder="ccache-$version-windows-x86_64"
$url = "https://github.com/ccache/ccache/releases/download/v$version/$folder.zip"
$expectedSha256 = "3D7CEBB05850AD704E197B3F1D3F0F924AB6C9FDFC561578E146184FE9D89380"

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $true

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
