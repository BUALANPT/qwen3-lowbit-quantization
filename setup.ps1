$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Utility -ErrorAction Stop
Import-Module Microsoft.PowerShell.Archive -ErrorAction Stop
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$ArgumentList)
    & $Program @ArgumentList
    if ($LASTEXITCODE -ne 0) { throw "$Program exited with code $LASTEXITCODE" }
}

function Get-Archive {
    param([string]$Url, [string]$Archive, [string]$Destination, [string[]]$RequiredFiles, [string]$Sha256)
    $missing = @($RequiredFiles | Where-Object { !(Test-Path -LiteralPath (Join-Path $Destination $_) -PathType Leaf) })
    if ($missing.Count -eq 0) { return }
    Invoke-Checked 'curl.exe' @('-fsSL','--retry','3','-o',"$Archive.part",$Url)
    if ($Sha256 -and (Get-FileHash -LiteralPath "$Archive.part" -Algorithm SHA256).Hash -ne $Sha256) {
        throw 'Downloaded archive does not match the pinned SHA-256.'
    }
    Move-Item -LiteralPath "$Archive.part" -Destination $Archive -Force
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    foreach ($file in $RequiredFiles) {
        if (!(Test-Path -LiteralPath (Join-Path $Destination $file) -PathType Leaf)) {
            throw "Archive did not contain $file"
        }
    }
}

New-Item -ItemType Directory -Force -Path 'tools','models/hf','data' | Out-Null
$release = 'https://github.com/ggml-org/llama.cpp/releases/download/b12100'
$programArchive = 'llama-b12100-bin-win-cuda-13.4-x64.zip'
$cudaArchive = 'cudart-llama-bin-win-cuda-13.4-x64.zip'
Get-Archive "$release/$programArchive" "tools/$programArchive" 'tools/llama' @(
    'llama-bench.exe','llama-completion.exe','llama-quantize.exe','llama-imatrix.exe','llama-perplexity.exe',
    'llama-bench-impl.dll','llama-completion-impl.dll','llama-quantize-impl.dll','llama-perplexity-impl.dll',
    'llama.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll','libomp.dll')
Get-Archive "$release/$cudaArchive" "tools/$cudaArchive" 'tools/llama' @(
    'cublas64_13.dll','cublasLt64_13.dll','cudart64_13.dll')

if (!(Test-Path -LiteralPath 'tools/source/convert_hf_to_gguf.py')) {
    Invoke-Checked 'git' @('-c','advice.detachedHead=false','clone','--depth','1','--branch','b12100',
        'https://github.com/ggml-org/llama.cpp.git','tools/source')
}
if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    Invoke-Checked 'py' @('-3.11','-m','venv','.venv')
}
Invoke-Checked '.\.venv\Scripts\python.exe' @('-m','pip','install','--disable-pip-version-check','--progress-bar','off',
    '-r','tools/source/requirements/requirements-convert_hf_to_gguf.txt',
    'gguf==0.19.0','safetensors==0.8.0','matplotlib==3.11.2')

$revision = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
$modelUrl = "https://huggingface.co/Qwen/Qwen3-1.7B-Base/resolve/$revision"
foreach ($name in @('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','LICENSE')) {
    if (!(Test-Path -LiteralPath "models/hf/$name")) {
        Invoke-Checked 'curl.exe' @('-fsSL','--retry','3','-o',"models/hf/$name.part","$modelUrl/$name")
        Move-Item -LiteralPath "models/hf/$name.part" -Destination "models/hf/$name"
    }
}
$weights = 'models/hf/model.safetensors'
$expectedHash = '6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841'
if (!(Test-Path -LiteralPath $weights)) {
    Invoke-Checked 'curl.exe' @('-fsSL','--retry','3','-C','-','-o',"$weights.part","$modelUrl/model.safetensors")
    if ((Get-FileHash -LiteralPath "$weights.part" -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'Downloaded checkpoint does not match the pinned SHA-256.'
    }
    Move-Item -LiteralPath "$weights.part" -Destination $weights
}
Get-Archive 'https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip' `
    'data/wikitext-2.zip' 'data' @('wikitext-2-raw/wiki.train.raw','wikitext-2-raw/wiki.valid.raw',
    'wikitext-2-raw/wiki.test.raw') 'ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11'

Invoke-Checked '.\tools\llama\llama-bench.exe' @('--list-devices')
