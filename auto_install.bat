@echo off
chcp 65001 >nul
title STTTS - Install GPT-SoVITS
cd /d "%~dp0"

echo ============================================
echo   STTTS - GPT-SoVITS 自动安装脚本
echo ============================================
echo.

REM ── Check prerequisites ──
where git >nul 2>&1 || (echo [ERROR] git not found. Install git first. & pause & exit /b 1)
where conda >nul 2>&1 || (echo [ERROR] conda not found. Install miniconda first. & pause & exit /b 1)

set GPT_DIR=GPT-SoVITS

REM ── Clone ──
if not exist "%GPT_DIR%" (
    echo [1/5] Cloning GPT-SoVITS...
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git "%GPT_DIR%"
) else (
    echo [1/5] GPT-SoVITS already cloned, updating...
    git -C "%GPT_DIR%" pull
)

REM ── Create conda env ──
echo [2/5] Creating conda environment (Python 3.10)...
conda create -n GPTSoVits python=3.10 -y

REM ── Install PyTorch with CUDA ──
echo [3/5] Installing PyTorch with CUDA 12.4 (this may take a while)...
conda run -n GPTSoVits pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM ── Install pip deps ──
echo [4/5] Installing Python dependencies...
cd "%GPT_DIR%"

REM Install core deps for api_v2.py (skip heavy training deps)
conda run -n GPTSoVits pip install fastapi uvicorn[standard] pydantic PyYAML soundfile librosa scipy tqdm
conda run -n GPTSoVits pip install transformers sentencepiece torchmetrics ctranslate2 av cn2an pypinyin chardet psutil jieba opencc-python-reimplemented
conda run -n GPTSoVits pip install peft gradio pytorch-lightning split-lang fast_langdetect rotary_embedding_torch x_transformers

cd "%~dp0"

REM ── Download pretrained models (v2 base) ──
echo [5/5] Downloading pretrained models (v2, ~3.5GB total, first run only)...
conda run -n GPTSoVits python -c "
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
"

echo.
echo ============================================
echo   Installation complete!
echo.
echo   Start GPT-SoVITS:
echo     conda activate GPTSoVits
echo     cd GPT-SoVITS
echo     python api_v2.py -a 127.0.0.1 -p 9880
echo.
echo   Or use the GUI (python gui.py) and click
echo   the "▶ GPT-SoVITS" button.
echo ============================================

echo.
echo --- Installing torchaudio for Silero VAD (into GPTSoVits conda env) ---
conda run -n GPTSoVits pip install torchaudio --quiet

echo.
echo --- Pre-downloading Whisper STT model (tiny) ---
set PYTHON=python
if exist "venv\Scripts\python.exe" set PYTHON=venv\Scripts\python.exe
%PYTHON% -c "import faster_whisper" 2>nul
if %errorlevel% neq 0 %PYTHON% -m pip install faster-whisper --quiet
%PYTHON% -c "
from faster_whisper import download_model
import os
model_dir = os.path.expanduser('~/.cache/faster-whisper/tiny')
if not os.path.exists(model_dir):
    print('Downloading faster-whisper tiny model (~150MB)...')
    download_model('tiny')
    print('Whisper model ready')
else:
    print('Whisper model already cached')
" 2>&1 || echo Whisper model download skipped
echo.
pause
