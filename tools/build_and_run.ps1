$vcvars = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
)

$vcvarsPath = $null
foreach ($path in $vcvars) {
    if (Test-Path $path) {
        $vcvarsPath = $path
        break
    }
}

if ($null -eq $vcvarsPath) {
    Write-Host "Could not find vcvars64.bat. Please run in VS Developer Command Prompt."
    exit 1
}

Write-Host "Found vcvars64.bat at: $vcvarsPath"

$slnPath = "BitCrack.sln"
if (!(Test-Path $slnPath)) {
    $slnPath = "..\BitCrack.sln"
    if (!(Test-Path $slnPath)) {
        Write-Host "Error: Could not find BitCrack.sln in current or parent directory."
        exit 1
    }
}
$buildCmd = "msbuild `"$slnPath`" /p:Configuration=Release /p:Platform=x64 /t:ECDump"
$cmd = "call `"$vcvarsPath`" && $buildCmd"

Write-Host "Building ECDump..."
cmd /c $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build Successful."
}
else {
    Write-Host "Build Failed."
    exit 1
}
