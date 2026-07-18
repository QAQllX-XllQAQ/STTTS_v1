# STTTS_v1

Speech-to-Text → Text-to-Speech 实时语音转换工具。macOS 版本。

## 快速开始

```bash
pip install -r requirements.txt
python3 gui.py
```

Windows 双击 `启动GUI.bat` · macOS 双击 `启动GUI.command`

## 功能

| 功能 | 选项 |
|---|---|
| STT | Whisper（本地 GPU/MPS）/ Google Cloud STT |
| TTS | GPT-SoVITS / Edge-TTS |
| 音频输出 | 任意输出设备 |
| GPT-SoVITS 管理 | GUI 内一键启动/停止 |

## 安装 GPT-SoVITS（可选）

**Windows:** 双击 `auto_install.bat`  
**macOS:** 终端运行 `bash auto_install.sh`

安装完成后，在 GUI 里点 **▶ GPT-SoVITS** 启动。

## 目录结构

```
STTTS_v1/
├── gui.py                 # 桌面 GUI（推荐）
├── STTTS.py               # 命令行版
├── 启动GUI.bat            # Windows 双击启动
├── 启动GUI.command        # macOS 双击启动
├── auto_install.bat       # Windows GPT-SoVITS 安装
├── auto_install.sh        # macOS GPT-SoVITS 安装
├── requirements.txt       # Python 依赖
├── RealtimeSTT/           # 语音识别库
└── README.md
```
