FROM python:3.12-slim

LABEL maintainer="Watermark Remover Pro Team"
LABEL description="AI-Powered Video/Image Watermark Removal - VSR Enhanced v2.0"
LABEL version="2.0.0"

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制 requirements.txt 以利用 Docker 缓存层
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装额外的 WebUI 依赖
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.34.0 \
    gradio==5.9.1 \
    python-multipart==0.0.20 \
    pydantic==2.10.4 \
    transformers>=4.46.0 \
    torch>=2.7.0 \
    torchvision>=0.22.0

# 复制项目代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/uploads /app/output /app/models

# 预下载 Florence-2 模型（可选，首次启动时也可自动下载）
# Bug #11 修复: 模型名从 Florence-2-base-ft 改为 Florence-2-base（HuggingFace 官方名称）
RUN python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)" 2>/dev/null || echo "Model will be downloaded on first use"

# 暴露端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# 启动命令
CMD ["python3", "webui.py", "--host", "0.0.0.0", "--port", "7860"]
