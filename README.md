# Video Remover

> 本地运行的 AI 视频字幕 / 水印去除工具

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org)
[![Platform](https://img.shields.io/badge/OS-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](#)
[![Docker](https://img.shields.io/badge/Docker-GHCR-blue?logo=docker)](https://ghcr.io/q249117652/video-remover)

---

## 这是什么

**Video Remover** 是一款完全在本地运行的桌面与命令行工具，利用多种深度学习修复模型自动识别并抹除视频中的**硬字幕**与**静态水印**，再用上下文修复算法补全被遮挡的画面区域，输出接近无痕的结果。

整个处理过程不依赖任何外部云服务，视频不会离开你的机器。

### 核心能力

- **硬字幕去除**：在尽量保持原始分辨率的前提下，抹除烧录在画面上的字幕文字
- **AI 上下文填充**：用 STTN / LAMA / ProPainter 等修复模型重建被遮挡区域，而非简单模糊或打马赛克
- **灵活的区域控制**：既可以传入坐标只清理指定区域，也可以全自动检测并清除整段视频中的文本
- **批量图片去水印**：同样的能力也适用于静态图片中的水印 / 文字
- **多后端加速**：支持 NVIDIA CUDA、AMD / Intel DirectML、Apple Silicon 以及纯 CPU 运行

---

## 处理流程

```
输入视频
   │
   ▼
字幕检测 (PP-OCR 文本检测)
   │
   ▼
生成遮罩区域
   │
   ▼
修复模型 (STTN / LAMA / ProPainter / OpenCV)
   │
   ▼
画面重建 + 音轨合并
   │
   ▼
输出视频
```

下面按阶段说明每一步实际在做什么：

### 1. 输入与解封装
视频文件先由 ffmpeg 解封装，分离出连续的视频帧序列与原始音轨。这一步只做拆解、不改动画面内容，音轨会被原样保留到最后的合成阶段。

### 2. 字幕 / 文本检测
对每一帧调用 PP-OCR 文本检测模型，定位画面中出现文字的矩形区域。你可以：
- **不传坐标** → 自动检测整帧中的所有文本；
- **传入坐标**（如 `-c 900 1080 0 1920`）→ 只在指定区域内检测，减少误伤。

检测结果会汇总成一份"需要清理的位置清单"。

### 3. 生成遮罩区域
把检测到的文本区域转换成二值遮罩（mask），标记出每一帧里"应当被修复填补"的像素。对于固定位置的字幕（如片尾滚动字幕），遮罩可在多帧之间复用，从而提升处理速度。

### 4. 画面修复（核心步骤）
修复模型依据遮罩，用周围上下文信息重建被字幕遮挡的原始画面：
- **STTN**：基于时序卷积，擅长真人实拍视频，速度快，可跳过检测阶段；
- **LAMA**：基于傅里叶卷积，对图片与动画类画面效果好；
- **ProPainter**：基于光流传播，适合运动剧烈的镜头，但显存占用大；
- **OpenCV**：轻量传统算法，作为兜底方案。

### 5. 帧重建与音轨合并
修复后的帧序列重新编码为视频流，再与第 1 步保留的原始音轨合并，得到带声音的成片。

### 6. 输出
按指定路径写出最终视频，分辨率与输入保持一致（不做缩放，保留原始清晰度）。

---

## 快速开始

### 方式一：Docker（推荐，开箱即用）

镜像已发布到 GitHub Container Registry，无需登录即可拉取：

```shell
# NVIDIA 10 / 20 / 30 系显卡（CUDA 11.8）
docker run -it --name vr --gpus all \
  ghcr.io/q249117652/video-remover:cuda-11.8 \
  python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# NVIDIA 40 系显卡（CUDA 12.6）
docker run -it --name vr --gpus all \
  ghcr.io/q249117652/video-remover:cuda-12.6 \
  python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# 无显卡 / CPU 模式
docker run -it --name vr \
  ghcr.io/q249117652/video-remover:cpu-latest \
  python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# 把处理好的视频拷贝出来
docker cp vr:/vsr/test/test_no_sub.mp4 ./
```

可用标签：`cuda-11.8` · `cuda-12.6` · `cpu-latest` · `main`

### 方式二：源码运行

由于仓库未包含大体积模型与二进制文件，请先从 Release 下载后再运行：

```shell
# 1. 克隆仓库
git clone https://github.com/q249117652/video-remover.git
cd video-remover

# 2. 从 Release 下载大文件（ffmpeg、AI 模型、素材）
#    下载地址：https://github.com/q249117652/video-remover/releases/tag/v1.0.0-assets
#    按下方表格放回对应目录

# 3. 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. 运行图形界面 / 命令行
python gui.py                   # 图形界面
python backend/main.py          # 命令行
```

#### Release 大文件与目录对应关系

| Release 资产前缀 | 内容 | 放回目录 |
|------------------|------|----------|
| `backend_ffmpeg_*` | ffmpeg 可执行文件（Windows / macOS / Linux） | `backend/ffmpeg/` |
| `backend_models_*` | AI 模型参数（STTN / LAMA / ProPainter / PP-OCR 等） | `backend/models/` |
| `design_*` | 图标、演示素材、参考资料 | `design/` |

> 说明：GitHub 对单文件体积与仓库体积有限制，因此上述大文件单独存放在 Release 中，源码保持轻量、易于克隆。

---

## 命令行参数

```text
Video Remover - Command Line Tool

options:
  -h, --help            显示帮助信息
  --input INPUT, -i     输入视频文件路径
  --output OUTPUT, -o   输出视频文件路径（可选）
  --subtitle-area-coords YMIN YMAX XMIN XMAX, -c
                        字幕区域坐标（可多次指定多个区域）
  --inpaint-mode {sttn-auto,sttn-det,lama,propainter,opencv}
                        修复模式，默认 sttn-auto
```

示例：

```shell
# 全自动去除整段视频字幕
python backend/main.py -i input.mp4 -o output.mp4

# 只清理画面底部指定区域
python backend/main.py -i input.mp4 -o output.mp4 -c 900 1080 0 1920
```

---

## 源码安装（按运行环境）

### NVIDIA CUDA（推荐）

```shell
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### CPU（无显卡）

```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch==2.7.0 torchvision==0.22.0
pip install -r requirements.txt
```

### DirectML（AMD / Intel 显卡）

```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt
pip install torch_directml==0.2.5.dev240914
```

### macOS（Apple Silicon）

```shell
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch==2.7.0 torchvision==0.22.0
pip install -r requirements.txt
```

---

## 调参与常见问题

### 处理太慢？

编辑 `backend/config.py`：

```python
MODE = InpaintMode.STTN          # 使用 STTN 算法
STTN_SKIP_DETECTION = True       # 跳过字幕检测（可能漏掉部分字幕）
```

### 效果不理想？

不同算法各有侧重：

- **STTN**：真人实拍视频效果好、速度快，可跳过检测
- **LAMA**：图片与动画类视频效果好，速度一般
- **ProPainter**：消耗显存大、速度慢，适合运动剧烈的镜头

调高参考帧数与邻域步长可提升质量（同时增加显存占用）：

```python
MODE = InpaintMode.STTN
STTN_NEIGHBOR_STRIDE = 10
STTN_REFERENCE_LENGTH = 10
STTN_MAX_LOAD_NUM = 30
```

### macOS 报错 "bad CPU type in executable"

执行 `softwareupdate --install-rosetta` 安装 Rosetta 后重试。

---

## 许可证与致谢

本项目以 **Apache License 2.0** 发布。

Video Remover 是基于开源项目 `video-subtitle-remover` 的派生版本，遵循 Apache 2.0 许可进行分发与修改。原始项目版权归原作者所有，本仓库保留其许可证与版权声明。

完整许可证文本见 [LICENSE](LICENSE)。
