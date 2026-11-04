param(
    [ValidateRange(1,20000)][int]$Steps = 10000,
    [string]$InitializeFrom
)
$ErrorActionPreference = 'Stop'
$workspaceDir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$experimentDir = Join-Path $workspaceDir '.work\act-public-pickplace-2026-10-29'
$pythonExe = Join-Path $experimentDir '.venv\Scripts\python.exe'
$cacheDir = Join-Path $experimentDir 'cache'
if (-not (Test-Path -LiteralPath $pythonExe) -or -not (Test-Path -LiteralPath (Join-Path $cacheDir 'manifest.json'))) {
    throw 'The private ACT environment or prepared dataset cache is missing. See README.md.'
}
$label = 'trial-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$runDir = Join-Path $PSScriptRoot $label
$guardReport = Join-Path $PSScriptRoot ($label + '-guard.json')
$env:CUDA_VISIBLE_DEVICES = '0'
$env:TORCH_HOME = Join-Path $experimentDir 'torch-cache'
$env:HF_HOME = Join-Path $experimentDir 'hf-cache'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
$trainArgs = @($pythonExe,'-u',(Join-Path $PSScriptRoot 'train_act.py'),'--cache',$cacheDir,
               '--out',$runDir,'--steps',[string]$Steps,'--batch','8','--eval-every','1000')
if ($InitializeFrom) {
    $checkpointDir = (Resolve-Path -LiteralPath $InitializeFrom).Path
    $trainArgs += @('--initialize-from',$checkpointDir)
}
& $pythonExe -u (Join-Path $PSScriptRoot 'guard_trial.py') --report $guardReport --seconds 1800 -- @trainArgs
if ($LASTEXITCODE -ne 0) { throw "ACT training failed or was stopped by its guard. Inspect $guardReport" }
Write-Output "ACT trial saved to $runDir"
