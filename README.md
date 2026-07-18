# STTTS_v1

Speech-to-Text → Text-to-Speech 实时语音转换工具。桌面 GUI，支持多引擎切换。

## 截图

```
┌──────────────────────────────────────────────┐
│ STT Engine                                    │
│  ● Whisper (local GPU)  ○ Google Cloud STT   │
│  Input device: [Voicemeeter Out B1       ▼]  │
├──────────────────────────────────────────────┤
│ TTS Engine                                    │
│  ● GPT-SoVITS  ○ Edge-TTS                    │
│  [▶ GPT-SoVITS] [■ GPT-SoVITS]  ✅ Running  │
├──────────────────────────────────────────────┤
│ Audio Output                                  │
│  Play to: [Headphones (Chu2 DSP)         ▼]  │
├──────────────────────────────────────────────┤
│ [▶ Start]  [■ Stop]                          │
├──────────────────────────────────────────────┤
│ 你 好                                         │
│ TTS OK (194KB)                               │
└──────────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 启动 GUI
python gui.py
```

或双击 `启动GUI.bat`。

## 安装 GPT-SoVITS（可选，用于高质量语音合成）

双击 `auto_install.bat`，一键完成：
- 克隆 GPT-SoVITS 仓库
- 创建 conda 环境 + 安装依赖
- 下载 v2 预训练模型（~3.5GB）

安装完成后，在 GUI 里点 **▶ GPT-SoVITS** 启动，或手动运行：

```bash
conda activate GPTSoVits
cd GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880
```

## 目录结构

```
STTTS_v1/
├── gui.py                 # 桌面 GUI（推荐）
├── STTTS.py               # 命令行版
├── 启动GUI.bat            # 双击启动 GUI
├── auto_install.bat       # 一键安装 GPT-SoVITS
├── requirements.txt       # Python 依赖
├── RealtimeSTT/           # 语音识别库（STT 引擎）
├── README.md
└── .gitignore
```

## 功能

| 功能 | 选项 |
|---|---|
| STT | Whisper（本地 GPU）/ Google Cloud STT |
| TTS | GPT-SoVITS / Edge-TTS |
| 音频输出 | 任意输出设备（耳机、扬声器、Voicemeeter 等） |
| GPT-SoVITS 管理 | GUI 内一键启动/停止 |
