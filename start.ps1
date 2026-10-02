$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create the .venv environment and install requirements.txt first. See README.md for setup.'
}
& $projectPython -m streamlit run app.py --server.address=127.0.0.1
exit $LASTEXITCODE
