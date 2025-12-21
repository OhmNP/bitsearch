$scriptDir = $PSScriptRoot
$exe = Join-Path $scriptDir "..\bin\bitcrack-ecdump.exe"
if (-not (Test-Path $exe)) { 
    # Fallback to current directory lookups if PSScriptRoot is empty or fails
    $exe = "..\bin\bitcrack-ecdump.exe"
    if (-not (Test-Path $exe)) {
        $exe = "bin\bitcrack-ecdump.exe"
        if (-not (Test-Path $exe)) {
            Write-Error "Exe not found at $exe or relative paths"; exit 1 
        }
    }
}

Write-Host "Running Endomorphism Test..."
& $exe --experiment endo --family consecutive --start-k 1 --batch-keys 1024 --batches 1 --out-bin test_endo.bin --telemetry test_endo.json

if ($LASTEXITCODE -eq 0) {
    Write-Host "Test Run Successful"
    if (Test-Path test_endo.bin) {
        $size = (Get-Item test_endo.bin).Length
        Write-Host "Output file size: $size bytes"
        if ($size -eq (1024 * 108)) {
            Write-Host "Size matches expected (1024 * 108)"
        }
        else {
            Write-Host "Size MISMATCH: Expected $(1024 * 108), Got $size"
        }
    }
}
else {
    Write-Host "Test Run Failed"
}
