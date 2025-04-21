[CmdletBinding()]
param (
    [Parameter(Mandatory = $true)]
    [string] $CudaMajorVersion,

    [Parameter(Mandatory = $true)]
    [string] $CudaMinorVersion
)

$version = "9.6.0.74"
$folder="cudnn-windows-x86_64-${version}_cuda${CudaMajorVersion}-archive"
$url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-${version}_cuda${CudaMajorVersion}-archive.zip"
$expectedSha256 = "65CA0F2D77A46DE1DEF35E289780B8D8729EF2FA39CF8DD0C8448E381DD2978C"

$ErrorActionPreference = "Stop"

try {
    Write-Host "Download cuDNN"
    $zipFilePath = Join-Path "$env:TEMP" "$folder.zip"
    Invoke-WebRequest -Uri $url -UseBasicParsing -OutFile "$zipFilePath" -MaximumRetryCount 3

    $hash = Get-FileHash $zipFilePath -Algorithm "sha256"
    if ($hash.Hash -ne $expectedSha256) {
        throw "File $Path hash $hash.Hash did not match expected hash $expectedHash"
    }

    Write-Host "Unzip cuDNN"
    Expand-Archive -Path "$zipFilePath" -DestinationPath "$env:TEMP" -Force
    
    Write-Host "Move cuDNN"
    $sourceFolder = "$env:TEMP/$folder"
    $targetFolder = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CudaMajorVersion}.${CudaMinorVersion}"
    Move-Item -Path "$sourceFolder/include/*" -Destination "$targetFolder/include" -Force
    Move-Item -Path "$sourceFolder/bin/*" -Destination "$targetFolder/bin" -Force
    Move-Item -Path "$sourceFolder/lib/x64/*" -Destination "$targetFolder/lib/x64" -Force
}
catch {
    Write-Host "Installation failed with an error"
    $_.Exception | Format-List
    exit -1
}
