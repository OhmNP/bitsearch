$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$AnalyzerPath = "$PSScriptRoot\crypto_analysis.py"
$PythonPath = "C:\Users\parim\AppData\Local\Programs\Python\Python313\python.exe"

$Files = Get-ChildItem -Path "$PSScriptRoot\..\phase3_2_*.bin"

foreach ($File in $Files) {
    $BinFile = $File.FullName
    $BaseName = $File.BaseName
    $AnalysisFile = "$PSScriptRoot\..\${BaseName}_analysis.json"
    
    Write-Host "Analyzing $BaseName..."
    & $PythonPath $AnalyzerPath $BinFile --output $AnalysisFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Analysis failed for $BaseName"
    }
    else {
        Write-Host "Success: $AnalysisFile"
    }
}
