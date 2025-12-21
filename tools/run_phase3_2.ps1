$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$ExePath = "$PSScriptRoot\..\bin\bitcrack-ecdump.exe"
$AnalyzerPath = "$PSScriptRoot\crypto_analysis.py"

function Run-Experiment {
    param (
        [string]$Name,
        [string]$Arguments,
        [int]$Keys
    )

    $BinFile = "phase3_2_${Name}.bin"
    $JsonFile = "phase3_2_${Name}.json"
    $AnalysisFile = "phase3_2_${Name}_analysis.json"
    
    Remove-Item $BinFile -ErrorAction SilentlyContinue

    Write-Host "Running Experiment: $Name ($Keys keys)"
    Write-Host "Command: $ExePath $Arguments --batch-keys $Keys --batches 1 --out-bin $BinFile --telemetry $JsonFile"
    
    $proc = Start-Process -FilePath $ExePath -ArgumentList "$Arguments --batch-keys $Keys --batches 1 --out-bin $BinFile --telemetry $JsonFile" -Wait -PassThru -NoNewWindow
    
    if ($proc.ExitCode -ne 0) {
        Write-Error "Experiment $Name failed with exit code $($proc.ExitCode)"
        return $false
    }
    
    Write-Host "Analyzing output..."
    # Explicitly use the python found by pip
    & "C:\Users\parim\AppData\Local\Programs\Python\Python313\python.exe" $AnalyzerPath $BinFile --output $AnalysisFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Analysis for $Name failed"
        return $false
    }
    
    Write-Host "Experiment $Name Complete via $AnalysisFile"
    return $true
}

$KeyCount = 1048576 # 1M keys

# 1. Control (Random)
Run-Experiment -Name "control" -Arguments "--family control" -Keys $KeyCount

# 2. Masked Scalars (High bits zeroed)
# Masks: 8, 16, 32, 64, 96, 128
$Masks = @(8, 16, 32, 64, 96, 128)
foreach ($mask in $Masks) {
    Run-Experiment -Name "masked_${mask}" -Arguments "--family masked --start-k 0x0 --mask-bits $mask" -Keys $KeyCount
}

# 3. Stride Scalars
# Strides: 2, 4, 8, 16, 256
$Strides = @(2, 4, 8, 16, 256)
foreach ($stride in $Strides) {
    # Start at random offset to avoid exact repetition if run multiple times
    $StartK = Get-Random -Minimum 1 -Maximum 1000000
    Run-Experiment -Name "stride_${stride}" -Arguments "--family stride --stride $stride --start-k $StartK" -Keys $KeyCount
}

Write-Host "All experiments completed."
