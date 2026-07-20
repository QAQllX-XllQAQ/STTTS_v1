# STTTS - GPT-SoVITS 自动安装脚本 (PowerShell)
# 需要管理员权限

# 自动提权
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
    Start-Process -FilePath "powershell.exe" -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs -WorkingDirectory "$PSScriptRoot"
    exit
}

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location "$PSScriptRoot"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  STTTS - GPT-SoVITS 自动安装脚本 (管理员)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] git not found. Install git first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] conda not found. Install miniconda first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$GPT_DIR = "GPT-SoVITS"

# Clone
if (-not (Test-Path $GPT_DIR)) {
    Write-Host "[1/5] Cloning GPT-SoVITS..." -ForegroundColor Yellow
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git $GPT_DIR
} else {
    Write-Host "[1/5] GPT-SoVITS already cloned, updating..." -ForegroundColor Yellow
    git -C $GPT_DIR pull
}

# Create conda env
Write-Host "[2/5] Creating conda environment (Python 3.10)..." -ForegroundColor Yellow
conda create -n GPTSoVits python=3.10 -y

# Install PyTorch with CUDA
Write-Host "[3/5] Installing PyTorch with CUDA 12.4 (this may take a while)..." -ForegroundColor Yellow
conda run -n GPTSoVits pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install pip deps
Write-Host "[4/5] Installing Python dependencies..." -ForegroundColor Yellow
Push-Location $GPT_DIR
conda run -n GPTSoVits pip install fastapi "uvicorn[standard]" pydantic PyYAML soundfile librosa scipy tqdm
conda run -n GPTSoVits pip install transformers sentencepiece torchmetrics ctranslate2 av cn2an pypinyin chardet psutil jieba opencc-python-reimplemented
conda run -n GPTSoVits pip install peft gradio pytorch-lightning split-lang fast_langdetect rotary_embedding_torch x_transformers
Pop-Location

# Download pretrained models
Write-Host "[5/5] Downloading pretrained models (v2, ~3.5GB total, first run only)..." -ForegroundColor Yellow
conda run -n GPTSoVits python -c @"
import huggingface_hub, os
repo = 'lj1995/GPT-SoVITS'
base = os.path.join(os.getcwd(), 'GPT-SoVITS', 'GPT_SoVITS', 'pretrained_models')
files = [
    'gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
    'gsv-v2final-pretrained/s2G2333k.pth',
    'chinese-roberta-wwm-ext-large/pytorch_model.bin',
    'chinese-roberta-wwm-ext-large/config.json',
    'chinese-roberta-wwm-ext-large/vocab.txt',
    'chinese-roberta-wwm-ext-large/tokenizer_config.json',
    'chinese-hubert-base/pytorch_model.bin',
    'chinese-hubert-base/config.json',
    'chinese-hubert-base/preprocessor_config.json',
]
os.makedirs(base, exist_ok=True)
for f in files:
    dst = os.path.join(base, f)
    if not os.path.exists(dst):
        print(f'Downloading {f}...')
        try:
            huggingface_hub.hf_hub_download(repo, f, local_dir=base, local_dir_use_symlinks=False)
        except Exception as e:
            print(f'  SKIP: {e}')
    else:
        print(f'  Already exists')
print('Done')
"@

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Start GPT-SoVITS:" -ForegroundColor White
Write-Host "    conda activate GPTSoVits"
Write-Host "    cd GPT-SoVITS"
Write-Host "    python api_v2.py -a 127.0.0.1 -p 9880"
Write-Host ""
Write-Host "  Or use the GUI (python gui.py) and click"
Write-Host "  the '▶ GPT-SoVITS' button."
Write-Host "============================================" -ForegroundColor Green

Write-Host ""
Write-Host "Installing torchaudio for Silero VAD..." -ForegroundColor Yellow
conda run -n GPTSoVits pip install torchaudio --quiet

Write-Host "Pre-downloading Whisper STT model (tiny)..." -ForegroundColor Yellow
$PYTHON = "python"
if (Test-Path "venv\Scripts\python.exe") { $PYTHON = "venv\Scripts\python.exe" }
& $PYTHON -c "import faster_whisper" 2>$null
if ($LASTEXITCODE -ne 0) { & $PYTHON -m pip install faster-whisper --quiet }
& $PYTHON -c @"
from faster_whisper import download_model
import os
model_dir = os.path.expanduser('~/.cache/faster-whisper/tiny')
if not os.path.exists(model_dir):
    print('Downloading faster-whisper tiny model (~150MB)...')
    download_model('tiny')
    print('Whisper model ready')
else:
    print('Whisper model already cached')
"@

Write-Host ""
Read-Host "Press Enter to exit"
