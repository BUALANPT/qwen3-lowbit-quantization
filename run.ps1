param(
    [ValidateSet('all','prepare','baselines','scan','evaluate','report','int4','test')]
    [string]$Stage = 'all'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = '.\.venv\Scripts\python.exe'

function Invoke-ProjectPython {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}

if ($Stage -in @('all','test')) { Invoke-ProjectPython -m unittest discover -s tests -v }
if ($Stage -in @('all','int4')) { Invoke-ProjectPython -u measure_int4.py }
if ($Stage -in @('all','prepare','baselines','scan','evaluate')) {
    Invoke-ProjectPython -u experiment.py $Stage
}
if ($Stage -in @('all','evaluate','report')) { Invoke-ProjectPython -u plot_results.py }
