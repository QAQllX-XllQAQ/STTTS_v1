# STTTS

Speech-to-Text → Text-to-Speech 实时语音转换工具。Windows 桌面 GUI。

## 快速开始

```bash
pip install -r requirements.txt
python gui.py
```

或双击 `启动GUI.bat`。

## 功能

| 功能 | 选项 |
|---|---|
| **STT** | Whisper（本地 GPU）/ Google Cloud STT（本地VAD / 全云端） |
| **TTS** | GPT-SoVITS / Edge-TTS / Google TTS |
| **音频输出** | 任意输出设备（耳机、扬声器、Voicemeeter 等） |
| **GPT-SoVITS** | GUI 内一键启动/停止 |

## 安装 GPT-SoVITS（可选）

双击 `auto_install.bat`，一键完成克隆 + conda 环境 + 下载模型。

安装后在 GUI 里点 **▶ GPT-SoVITS** 启动。

## 目录结构

```
STTTS/
├── gui.py                 # 桌面 GUI（推荐）
├── STTTS.py               # 命令行版
├── 启动GUI.bat            # 双击启动
├── auto_install.bat       # 一键安装 GPT-SoVITS
├── requirements.txt       # Python 依赖
├── RealtimeSTT/           # 语音识别库
└── README.md
```
