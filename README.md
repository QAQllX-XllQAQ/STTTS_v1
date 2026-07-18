# STTTS

Speech-to-Text → Text-to-Speech 实时语音转换工具。macOS 桌面 GUI。

## 快速开始

```bash
pip install -r requirements.txt
python3 gui.py
```

或双击 `启动GUI.command`。

## 功能

| 功能 | 选项 |
|---|---|
| **STT** | Whisper（本地 MPS）/ Google Cloud STT（本地VAD / 全云端） |
| **TTS** | GPT-SoVITS / Edge-TTS / Google TTS |
| **音频输出** | 任意输出设备 |
| **GPT-SoVITS** | GUI 内一键启动/停止 |

## 安装 GPT-SoVITS（可选）

```bash
bash auto_install.sh
```

安装后在 GUI 里点 **▶ GPT-SoVITS** 启动。

## 目录结构

```
STTTS/
├── gui.py                 # 桌面 GUI（推荐）
├── STTTS.py               # 命令行版
├── 启动GUI.command        # 双击启动
├── auto_install.sh        # 一键安装 GPT-SoVITS
├── requirements.txt       # Python 依赖
├── RealtimeSTT/           # 语音识别库
└── README.md
```
