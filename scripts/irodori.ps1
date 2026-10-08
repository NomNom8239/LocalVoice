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

# Windows PowerShell can promote native stderr to an error under "Stop".
# uv legitimately reports progress on stderr, so use Continue only while
# executing native programs, and check their exit codes explicitly.
function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "$FailureMessage (exit $exitCode)"
    }
}

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
        Invoke-NativeChecked -Executable "uv" -Arguments @("venv", $envDir, "--python", "3.12") -FailureMessage "ASR environment creation failed"
    }
    # Missing faster-whisper is expected in a newly created environment.
    # find_spec checks availability without importing and writing traceback.
    $probe = "import importlib.util; print('installed' if importlib.util.find_spec('faster_whisper') else 'missing')"
    $packageState = & $python -c $probe
    if ($LASTEXITCODE -ne 0) { throw "ASR dependency probe failed" }
    if ($packageState -eq "missing") {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv is needed to install faster-whisper."
        }
        Invoke-NativeChecked -Executable "uv" -Arguments @("pip", "install", "--python", $python, "faster-whisper") -FailureMessage "ASR dependency install failed"
    }
    elseif ($packageState -ne "installed") {
        throw "Unexpected ASR dependency probe result: $packageState"
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

Invoke-NativeChecked -Executable $python -Arguments $argsList -FailureMessage "Irodori $Action failed"
