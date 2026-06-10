<#
.SYNOPSIS
    Initialize the deepseismic2 local development environment.

.DESCRIPTION
    1. Verifies Docker Desktop is running.
    2. Starts Azurite (blob/queue/table emulator) via docker compose.
    3. Creates the required blob containers: raw, staged, features, results, catalog.
    4. Generates a synthetic ~5 MB SEG-Y test file using numpy (no segyio required).
    5. Uploads the sample file to the raw/ container.
    6. Prints next steps.

.PARAMETER ComposeFile
    Path to docker-compose.yml. Defaults to docker/docker-compose.yml relative to repo root.

.PARAMETER SkipSampleData
    Skip synthetic SEG-Y generation and upload.

.EXAMPLE
    # From the repo root:
    .\scripts\setup-local.ps1

.EXAMPLE
    .\scripts\setup-local.ps1 -SkipSampleData
#>
param (
    [string]$ComposeFile  = (Join-Path $PSScriptRoot "..\docker\docker-compose.yml"),
    [switch]$SkipSampleData
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Azurite well-known dev credentials — key is the standard emulator key
# See https://learn.microsoft.com/azure/storage/common/storage-use-azurite
$AzuriteKey = [System.Text.Encoding]::UTF8.GetString(
    [System.Convert]::FromBase64String(
        "RWJ5OHZkTTAyeE5PY3FGbHFVd0pQTGxtRXRsQ0RYSjFPY0hQa3pWMWtwU3ZCM1prU3ZKRlJYR3BBaVBNMVl3N0VqQ3E3VlhxMVVHQ3Y3WkJ2aGlTYkY9PQ=="
    )
)
$AzuriteConnStr = (
    "DefaultEndpointsProtocol=http;" +
    "AccountName=devstoreaccount1;" +
    "AccountKey=$AzuriteKey;" +
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)

$Containers = @("raw", "staged", "features", "results", "catalog")

function Write-Step { param($Msg) Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-OK   { param($Msg) Write-Host "    [OK] $Msg" -ForegroundColor Green }
function Write-Warn { param($Msg) Write-Host "    [!!] $Msg" -ForegroundColor Yellow }

# --------------------------------------------------------------------------- #
# 1. Docker check                                                              #
# --------------------------------------------------------------------------- #
Write-Step "Checking Docker..."
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
    Write-OK "Docker is running."
}
catch {
    Write-Error "Docker is not running or not installed.`nStart Docker Desktop then retry."
    exit 1
}

# --------------------------------------------------------------------------- #
# 2. Start Azurite                                                             #
# --------------------------------------------------------------------------- #
Write-Step "Starting Azurite (storage emulator)..."
docker compose -f $ComposeFile up -d azurite
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose failed. Check $ComposeFile."
    exit 1
}

# Wait for Azurite blob port (10000)
Write-Host "    Waiting for Azurite blob port 10000 " -NoNewline -ForegroundColor Gray
$ready  = $false
$max    = 30
for ($i = 0; $i -lt $max; $i++) {
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", 10000)
        $tcp.Close()
        $ready = $true
        break
    }
    catch {
        Write-Host "." -NoNewline -ForegroundColor Gray
        Start-Sleep -Milliseconds 500
    }
}
Write-Host ""
if (-not $ready) {
    Write-Error "Azurite did not become available within $($max/2) seconds."
    exit 1
}
Write-OK "Azurite is ready on port 10000."

# --------------------------------------------------------------------------- #
# 3. Create blob containers                                                    #
# --------------------------------------------------------------------------- #
Write-Step "Creating blob containers..."
$env:STORAGE_CONNECTION_STRING = $AzuriteConnStr

$createScript = @'
import os, sys
from azure.storage.blob import BlobServiceClient
svc = BlobServiceClient.from_connection_string(os.environ["STORAGE_CONNECTION_STRING"])
for name in sys.argv[1:]:
    try:
        svc.create_container(name)
        print(f"  created : {name}")
    except Exception:
        print(f"  exists  : {name}")
'@

python -c $createScript @Containers
if ($LASTEXITCODE -ne 0) {
    Write-Error "Container creation failed.  Is azure-storage-blob installed?"
    exit 1
}
Write-OK "Containers ready: $($Containers -join ', ')"

# --------------------------------------------------------------------------- #
# 4 & 5. Sample data                                                           #
# --------------------------------------------------------------------------- #
if (-not $SkipSampleData) {
    $RepoRoot  = Resolve-Path (Join-Path $PSScriptRoot "..")
    $DataDir   = Join-Path $RepoRoot "data"
    $SamplePath = Join-Path $DataDir "sample_synthetic.segy"

    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir | Out-Null
    }

    Write-Step "Generating synthetic SEG-Y test file (~5 MB)..."

    # Pure-numpy synthetic SEG-Y writer — no segyio required at this stage.
    # Format: 3200-byte text header + 400-byte binary header +
    #         N traces of (240-byte trace header + float32 samples).
    $genScript = @'
import struct, os, sys
import numpy as np

output       = sys.argv[1]
n_traces     = 500
n_samples    = 1000
sample_us    = 2000     # 2 ms sample interval

text_header  = (b"C deepseismic2 synthetic test data" + b" " * 3166)[:3200]

bin_hdr = bytearray(400)
struct.pack_into(">i", bin_hdr,  0, n_traces)       # job id
struct.pack_into(">h", bin_hdr, 16, n_samples)      # samples per trace
struct.pack_into(">h", bin_hdr, 18, sample_us)      # sample interval (us)
struct.pack_into(">h", bin_hdr, 24, 5)              # data format: IEEE float32

rng = np.random.default_rng(42)

with open(output, "wb") as f:
    f.write(text_header)
    f.write(bytes(bin_hdr))
    for i in range(n_traces):
        th = bytearray(240)
        struct.pack_into(">i", th,   0, i + 1)      # trace seq number
        struct.pack_into(">h", th, 114, n_samples)  # samples per trace
        struct.pack_into(">h", th, 116, sample_us)  # sample interval
        f.write(bytes(th))
        trace = rng.standard_normal(n_samples).astype(">f4")
        f.write(trace.tobytes())

size_mb = os.path.getsize(output) / 1024**2
print(f"  generated: {output}  ({size_mb:.1f} MB)")
'@

    python -c $genScript $SamplePath
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Could not generate sample data (numpy not installed?).  Skipping upload."
    }
    else {
        Write-Step "Uploading sample SEG-Y to raw/synthetic/..."

        $uploadScript = @'
import os, sys
from azure.storage.blob import BlobServiceClient
svc  = BlobServiceClient.from_connection_string(os.environ["STORAGE_CONNECTION_STRING"])
src, blob_name = sys.argv[1], sys.argv[2]
with open(src, "rb") as fh:
    svc.get_blob_client("raw", blob_name).upload_blob(fh, overwrite=True)
print(f"  uploaded raw/{blob_name}")
'@

        python -c $uploadScript $SamplePath "synthetic/sample_synthetic.segy"
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Upload failed — check Azurite logs in docker/volumes/azurite-data."
        }
        else {
            Write-OK "Sample data → raw/synthetic/sample_synthetic.segy"
        }
    }
}

# --------------------------------------------------------------------------- #
# Done                                                                         #
# --------------------------------------------------------------------------- #
$sep = "=" * 52
Write-Host "`n$sep" -ForegroundColor Green
Write-Host "  deepseismic2 local dev environment is ready!" -ForegroundColor Green
Write-Host $sep -ForegroundColor Green
Write-Host @"

  Azurite endpoints:
    Blob   http://127.0.0.1:10000
    Queue  http://127.0.0.1:10001
    Table  http://127.0.0.1:10002

  Containers created: $($Containers -join ', ')

  Next steps:
    1. Copy the env template:
         cp .env.example .env
       (defaults already point to Azurite — no edits needed for local dev)

    2. Install the project (editable):
         pip install -e ".[dev]"

    3. Run tests:
         pytest src/tests

    4. Start the API:
         uvicorn deepseismic.api.main:app --reload

  To stop Azurite:
    docker compose -f docker/docker-compose.yml down

"@
