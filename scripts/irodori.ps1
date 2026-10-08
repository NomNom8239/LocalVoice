param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("scan", "status", "asr", "merge", "triage", "review", "classify", "export")]
    [string]$Action,

    [Parameter(Mandatory = $true)]
    [string]$Speaker,

    [int]$Limit = 30,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [ValidateSet("normal", "whisper", "laugh", "breath", "panting", "groan", "unknown")]
    [string]$Candidate,
    [ValidateSet("emotion", "other", "all")]
    [string]$Kind = "emotion",
    [ValidateSet("short_audio", "no_detected_text", "expressive_or_unclear", "short_transcript", "ordinary_candidate", "invalid_audio", "tagged_nonverbal", "needs_transcript")]
    [string]$Group,
    [switch]$NoPlay,
    [switch]$IncludeTagged,
    [switch]$RetryErrors,
    [switch]$IncludeShort,
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
    # faster-whisper 1.2.1 calls av.open(metadata_errors=...), which
    # PyAV 19 removed. Keep a tested compatible ASR pair, including repairs
    # for pre-existing .venv-asr installations.
    $probe = "import importlib.util as u, importlib.metadata as m; a=u.find_spec('av'); f=u.find_spec('faster_whisper'); print((m.version('faster-whisper') if f else 'missing') + '|' + (m.version('av') if a else 'missing'))"
    $packageState = & $python -c $probe
    if ($LASTEXITCODE -ne 0) { throw "ASR dependency version probe failed" }
    if ($packageState -ne "1.2.1|18.1.0") {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv is needed to install compatible ASR dependencies."
        }
        Write-Host "Installing compatible ASR dependencies (faster-whisper 1.2.1 / PyAV 18.1.0)..."
        Invoke-NativeChecked -Executable "uv" -Arguments @(
            "pip", "install", "--python", $python,
            "faster-whisper==1.2.1", "av==18.1.0"
        ) -FailureMessage "ASR dependency compatibility repair failed"
        $packageState = & $python -c $probe
        if ($LASTEXITCODE -ne 0 -or $packageState -ne "1.2.1|18.1.0") {
            throw "ASR dependencies remain incompatible: $packageState"
        }
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
    if ($IncludeShort) { $argsList += "--include-short" }
}
if ($Action -eq "review") {
    $argsList += @("--kind", $Kind, "--limit", [string]$Limit)
    if ($Group) { $argsList += @("--group", $Group) }
    if ($NoPlay) { $argsList += "--no-play" }
    if ($IncludeTagged) { $argsList += "--include-tagged" }
    if ($Candidate) { $argsList += @("--candidate", $Candidate) }
}
if ($Action -eq "classify") {
    $classifierDriver = Join-Path $PSScriptRoot "irodori_style.py"
    $argsList = @($classifierDriver, "--profile", $Speaker,
                  "--limit", [string]$Limit, "--device", $Device)
    if ($Workspace) { $argsList += @("--workspace", $Workspace) }
    if ($Group) { $argsList += @("--group", $Group) }
    if ($RetryErrors) { $argsList += "--retry-errors" }
}
if ($Action -eq "export" -and $Replace) { $argsList += "--replace" }

Invoke-NativeChecked -Executable $python -Arguments $argsList -FailureMessage "Irodori $Action failed"
