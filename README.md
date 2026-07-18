# STTTS_v1

Speech-to-Text → Text-to-Speech 实时语音转换工具。

## 功能

- **STT**: Whisper (本地 GPU) / Google Cloud STT
- **TTS**: GPT-SoVITS / Edge-TTS
- **音频输出**: 可选任意输出设备
- **界面**: GUI 桌面窗口 (PySimpleGUI) / 命令行模式

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 GUI
python gui.py
```

或双击 `启动GUI.bat`。

## 目录结构

```
STTTS_v1/
├── gui.py            # 桌面 GUI（推荐）
├── STTTS.py           # 命令行版
├── 启动GUI.bat        # 双击启动 GUI
├── requirements.txt   # Python 依赖
├── RealtimeSTT/       # 语音识别库（STT 引擎）
└── .gitignore
```

## 可选依赖

- **GPT-SoVITS**: 单独克隆 https://github.com/RVC-Boss/GPT-SoVITS，启动 api_v2.py
- **Google Cloud STT**: 需要 API key（GUI 中填写）
- **Whisper 模型**: 首次使用自动下载（~75MB tiny 或 ~3GB large-v2）

