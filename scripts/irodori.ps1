param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("scan", "status", "asr", "merge", "export")]
    [string]$Action,

    [Parameter(Mandatory = $true)]
    [string]$Speaker,

    [int]$Limit = 30,
    [switch]$RetryErrors,
    [switch]$Replace,
    [string]$Workspace
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$driver = Join-Path $PSScriptRoot "irodori_dataset.py"
$python = Join-Path $root ".venv\Scripts\python.exe"

if ($Action -eq "asr") {
    # Keep faster-whisper and CTranslate2 outside the CUDA/RVC environments.
    $envDir = Join-Path $root ".venv-asr"
    $python = Join-Path $envDir "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv is needed to bootstrap the isolated ASR environment."
        }
        & uv venv $envDir --python 3.12
        if ($LASTEXITCODE -ne 0) { throw "ASR environment creation failed" }
    }
    & $python -c "import faster_whisper" *> $null
    if ($LASTEXITCODE -ne 0) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv is needed to install faster-whisper."
        }
        & uv pip install --python $python faster-whisper
        if ($LASTEXITCODE -ne 0) { throw "ASR dependency install failed" }
    }
}
elseif (-not (Test-Path -LiteralPath $python)) {
    throw "LocalVoice .venv was not found: $python"
}

$argsList = @($driver, "--profile", $Speaker)
if ($Workspace) { $argsList += @("--workspace", $Workspace) }
$argsList += $Action

if ($Action -eq "asr") {
    $argsList += @("--limit", [string]$Limit)
    if ($RetryErrors) { $argsList += "--retry-errors" }
}
if ($Action -eq "export" -and $Replace) { $argsList += "--replace" }

& $python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Irodori $Action failed (exit $LASTEXITCODE)"
}
