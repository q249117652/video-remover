# Video Remover

> A local, AI-powered tool for removing subtitles and watermarks from video.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org)
[![Platform](https://img.shields.io/badge/OS-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](#)
[![Docker](https://img.shields.io/badge/Docker-GHCR-blue?logo=docker)](https://ghcr.io/q249117652/video-remover)

---

## What it does

**Video Remover** is a desktop and command-line tool that runs entirely on your own machine. It uses deep-learning inpainting models to detect and erase **burned-in subtitles** and **static watermarks** from video, then reconstructs the occluded regions with contextual filling so the result looks clean rather than blurred or pixelated.

Nothing leaves your computer — no cloud, no upload.

### Highlights

- **Hard-subtitle removal** at (near) original resolution
- **AI context filling** via STTN / LAMA / ProPainter instead of naive blur or mosaic
- **Flexible region control** — specify coordinates or let it auto-detect all text
- **Batch image watermark removal** with the same engine
- **Multiple backends** — NVIDIA CUDA, AMD / Intel DirectML, Apple Silicon, and pure CPU

---

## Pipeline

```
Input video
   │
   ▼
Subtitle detection (PP-OCR text detector)
   │
   ▼
Build mask regions
   │
   ▼
Inpaint model (STTN / LAMA / ProPainter / OpenCV)
   │
   ▼
Frame reconstruction + audio mux
   │
   ▼
Output video
```

---

## Quick start

### Option 1: Docker (recommended)

Images are published to GitHub Container Registry and can be pulled without login:

```shell
# NVIDIA 10 / 20 / 30 series (CUDA 11.8)
docker run -it --name vr --gpus all \
  ghcr.io/q249117652/video-remover:cuda-11.8 \
  python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# NVIDIA 40 series (CUDA 12.6)
docker run -it --name vr --gpus all \
  ghcr.io/q249117652/video-remover:cuda-12.6 \
  python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# CPU only
docker run -it --name vr \
  ghcr.io/q249117652/video-remover:cpu-latest \
  python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# Copy the result out of the container
docker cp vr:/vsr/test/test_no_sub.mp4 ./
```

Tags available: `cuda-11.8` · `cuda-12.6` · `cpu-latest` · `main`

### Option 2: Run from source

Large model and binary files are not stored in the repo. Download them from the Release first:

```shell
# 1. Clone
git clone https://github.com/q249117652/video-remover.git
cd video-remover

# 2. Download large files from the Release page
#    https://github.com/q249117652/video-remover/releases/tag/v1.0.0-assets
#    Place them back into the directories shown below

# 3. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Launch the GUI or CLI
python gui.py                   # GUI
python backend/main.py          # CLI
```

#### Release assets → directory map

| Asset prefix    | Contents                              | Target directory |
|-----------------|---------------------------------------|------------------|
| `backend_ffmpeg_*`  | ffmpeg binaries (Windows / macOS / Linux) | `backend/ffmpeg/` |
| `backend_models_*`  | AI model weights (STTN / LAMA / ProPainter / PP-OCR) | `backend/models/` |
| `design_*`          | icons, demo assets, references        | `design/` |

> Note: GitHub limits single-file and repo size, so the large assets live in the Release while the source stays lightweight and easy to clone.

---

## Command-line options

```text
Video Remover - Command Line Tool

options:
  -h, --help            Show this help message
  --input INPUT, -i     Input video file path
  --output OUTPUT, -o   Output video file path (optional)
  --subtitle-area-coords YMIN YMAX XMIN XMAX, -c
                        Subtitle area coordinates (repeatable)
  --inpaint-mode {sttn-auto,sttn-det,lama,propainter,opencv}
                        Inpaint mode, default sttn-auto
```

Examples:

```shell
# Auto-remove subtitles from the whole video
python backend/main.py -i input.mp4 -o output.mp4

# Only clean a specific bottom region
python backend/main.py -i input.mp4 -o output.mp4 -c 900 1080 0 1920
```

---

## Build from source (by environment)

### NVIDIA CUDA (recommended)

```shell
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### CPU (no GPU)

```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch==2.7.0 torchvision==0.22.0
pip install -r requirements.txt
```

### DirectML (AMD / Intel GPU)

```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt
pip install torch_directml==0.2.5.dev240914
```

### macOS (Apple Silicon)

```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch==2.7.0 torchvision==0.22.0
pip install -r requirements.txt
```

---

## Tuning & FAQ

### Too slow?

Edit `backend/config.py`:

```python
MODE = InpaintMode.STTN          # use STTN
STTN_SKIP_DETECTION = True       # skip detection (may miss some subtitles)
```

### Quality not good enough?

- **STTN**: great for real footage, fast, can skip detection
- **LAMA**: best for images and animated video, moderate speed
- **ProPainter**: heavy on VRAM, slow, good for fast motion

Increase reference length / neighbor stride for better quality (more VRAM):

```python
MODE = InpaintMode.STTN
STTN_NEIGHBOR_STRIDE = 10
STTN_REFERENCE_LENGTH = 10
STTN_MAX_LOAD_NUM = 30
```

### macOS "bad CPU type in executable"

Run `softwareupdate --install-rosetta` to install Rosetta, then retry.

---

## License & Credits

This project is released under the **Apache License 2.0**.

Video Remover is a derivative work based on the open-source project `video-subtitle-remover`, redistributed and modified under the Apache 2.0 license. Copyright of the original project belongs to its authors; this repository retains the original license and copyright notices.

See [LICENSE](LICENSE) for the full text.
