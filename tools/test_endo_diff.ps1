$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$ExePath = "$PSScriptRoot\..\bin\bitcrack-ecdump.exe"

Remove-Item "test_endo_diff.bin" -ErrorAction SilentlyContinue

Write-Host "Running ECDump with --experiment endo_diff..."
& $ExePath --family consecutive --start-k 1 --batch-keys 256 --batches 1 --experiment endo_diff --delta-set 1 --out-bin "test_endo_diff.bin" --telemetry "test_endo_diff.json"

if ($LastExitCode -eq 0) {
    Write-Host "ECDump finished successfully."
    Write-Host "Verifying output with verify_endo_diff.py..."
    python "$PSScriptRoot\verify_endo_diff.py" "test_endo_diff.bin"
}
else {
    Write-Host "ECDump failed."
    exit 1
}
