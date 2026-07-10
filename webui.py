"""
WebUI 服务端 - 基于 FastAPI + Gradio 的 Web 界面

提供 Docker 部署友好的 Web 界面，功能包括：
  - 视频上传与预览
  - 水印检测参数配置
  - 自动检测结果可视化 + 人工修正
  - 修复算法选择与参数调优
  - 实时处理进度展示
  - 下载处理后文件

技术栈：
  - FastAPI: REST API 后端
  - Gradio: 前端交互界面（内嵌或独立运行）
  - uvicorn: ASGI 服务器

部署方式：
  - Docker: docker build -t watermark-remover-web . && docker run -p 7860:7860 ...
  - 直接: python webui.py
"""

import logging
import asyncio
import json
import tempfile
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import cv2

# FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Gradio
import gradio as gr

logger = logging.getLogger(__name__)

# ============================================================
# 数据模型
# ============================================================

class DetectionConfig(BaseModel):
    """水印检测配置。"""
    detector_type: str = Field(
        default="hybrid",
        description="检测器类型: paddleocr | florence2 | hybrid | manual",
    )
    detection_prompt: str = Field(default="watermark", description="Florence-2 检测提示词")
    max_bbox_percent: float = Field(default=10.0, ge=1, le=50)
    confidence_threshold: float = Field(default=0.3, ge=0.05, le=1.0)
    enable_manual_correction: bool = Field(default=True)
    detection_skip: int = Field(default=5, ge=1, le=30)
    fade_in: float = Field(default=0.0, ge=0, le=10)
    fade_out: float = Field(default=0.0, ge=0, le=10)

class InpaintConfig(BaseModel):
    """修复算法配置。"""
    inpaint_mode: str = Field(
        default="sttn-auto",
        description="修复模式: lama | sttn-auto | sttn-det | propainter | opencv",
    )
    lama_super_fast: bool = Field(default=False)
    sttn_neighbor_stride: int = Field(default=10, ge=1, le=50)
    sttn_reference_length: int = Field(default=10, ge=1, le=50)
    sttn_max_load_num: int = Field(default=30, ge=10, le=100)

class ProcessRequest(BaseModel):
    """完整处理请求。"""
    input_filename: str
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    inpaint: InpaintConfig = Field(default_factory=InpaintConfig)
    output_format: str = Field(default="mp4", pattern="^(mp4|avi|mov|webm|png|jpg|webp)$")

class ProcessResponse(BaseModel):
    """处理响应。"""
    success: bool
    message: str
    output_file: Optional[str] = None
    preview_image: Optional[str] = None  # base64 编码的预览图
    processing_time_seconds: Optional[float] = None


# ============================================================
# 全局状态管理
# ============================================================

class AppState:
    """应用全局状态。"""

    MAX_UPLOAD_SIZE_MB = 500  # 最大上传文件大小（MB）

    def __init__(self):
        self.upload_dir = Path(tempfile.mkdtemp(prefix="wmr_upload_"))
        self.output_dir = Path(tempfile.mkdtemp(prefix="wmr_output_"))
        self.processing_tasks: Dict[str, Dict[str, Any]] = {}
        self.detector = None
        self.inpainters = {}
        self._file_index: Dict[str, Path] = {}  # file_id → path 的快速索引（Bug #8 修复）

        logger.info(f"AppState 初始化: upload={self.upload_dir}, output={self.output_dir}")

    def cleanup(self):
        """清理临时文件。"""
        import shutil
        for d in [self.upload_dir, self.output_dir]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理。"""
    logger.info("WebUI 启动中...")
    yield
    logger.info("WebUI 关闭，清理资源...")
    state.cleanup()


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Watermark Remover Pro",
    description="AI-Powered Video/Image Watermark Removal Tool (VSR Enhanced)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# API 端点
# ------------------------------------------------------------

@app.get("/")
async def root():
    """根路径：返回 API 信息。"""
    return {
        "name": "Watermark Remover Pro",
        "version": "2.0.0",
        "endpoints": {
            "/docs": "Swagger API 文档",
            "/gradio": "Gradio WebUI 界面",
        },
    }


@app.get("/health")
async def health_check():
    """健康检查端点（用于 Docker 就绪探针）。"""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传视频/图片文件。

    返回文件 ID 和基本信息。
    """
    # 验证文件扩展名
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".png", ".jpg", ".jpeg", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    # 文件大小限制检查（Bug #10 修复：在读取前先检查 Content-Length）
    content_length = getattr(file, 'headers', {}).get('content-length', None)
    if content_length and int(content_length) > AppState.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大：{int(content_length)/(1024*1024):.1f}MB > {AppState.MAX_UPLOAD_SIZE_MB}MB",
        )

    # 保存文件
    file_id = f"{Path(file.filename).stem}_{os.urandom(4).hex()}"
    save_path = state.upload_dir / f"{file_id}{ext}"

    content = await file.read()

    # 二次校验实际大小（防止 Content-Length 被伪造）
    if len(content) > AppState.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（实际）：{len(content)/(1024*1024):.1f}MB > {AppState.MAX_UPLOAD_SIZE_MB}MB",
        )

    # 更新文件索引（Bug #8 修复）
    state._file_index[file_id] = save_path
    with open(save_path, "wb") as f:
        f.write(content)

    # 获取视频/图片信息
    is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webp"}
    info = _get_media_info(save_path, is_video)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(save_path),
        "size_bytes": len(content),
        "media_info": info,
    }


@app.post("/api/detect")
async def detect_watermarks(
    file_id: str = Form(...),
    config_json: str = Form(...),
):
    """
    对已上传文件执行水印检测。

    返回检测结果（边界框列表）和可视化预览图（base64）。
    """
    config = DetectionConfig.model_validate_json(config_json)

    # 查找文件
    file_path = _find_uploaded_file(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_id}")

    # 执行检测
    try:
        from backend.tools.watermark_detect import UnifiedWatermarkDetector

        detector = UnifiedWatermarkDetector(
            detector_type=config.detector_type,
            florence_config={
                "detection_prompt": config.detection_prompt,
                "max_bbox_percent": config.max_bbox_percent,
                "confidence_threshold": config.confidence_threshold,
            },
            enable_manual_correction=False,  # API 模式下不做阻塞式 GUI 修正
        )

        # 读取首帧
        is_video = file_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        if is_video:
            cap = cv2.VideoCapture(str(file_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise HTTPException(status_code=400, detail="无法读取视频帧")
        else:
            frame = cv2.imread(str(file_path))

        detections, wm_type = detector.detect_frame(frame)

        # 生成可视化预览
        vis_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{det["label"]} {det.get("confidence", 0):.0%}'
            cv2.putText(vis_frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Bug #3 修复: cv2.encode → cv2.imencode
        _, buf = cv2.imencode(".png", vis_frame)
        import base64
        preview_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "success": True,
            "detections": detections,
            "watermark_type": wm_type,
            "preview_image": preview_b64,
            "detection_count": len(detections),
        }

    except Exception as e:
        logger.error(f"检测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/api/process")
async def process_file(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    启动后台处理任务。

    支持同步（小文件/快速模式）和异步（大文件）两种方式。
    """
    task_id = os.urandom(8).hex()

    # 将任务加入后台队列
    background_tasks.add_task(_run_processing_task, task_id, request)

    state.processing_tasks[task_id] = {
        "status": "queued",
        "request": request.model_dump(),
        "progress": 0,
        "result": None,
    }

    return {"task_id": task_id, "status": "queued"}


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态和进度。"""
    if task_id not in state.processing_tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return state.processing_tasks[task_id]


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """下载处理结果文件。"""
    if task_id not in state.processing_tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    task = state.processing_tasks[task_id]
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    output_file = task.get("result", {}).get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="输出文件不存在")

    return FileResponse(output_file, filename=Path(output_file).name)


# ------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------

def _find_uploaded_file(file_id: str) -> Optional[Path]:
    """根据 file_id 查找已上传文件（优先使用索引，回退遍历）。"""
    # Bug #8 修复：优先使用 O(1) 字典查找
    if file_id in state._file_index:
        p = state._file_index[file_id]
        if p.exists() and p.is_file():
            return p
        else:
            # 索引失效，清理并回退到遍历
            del state._file_index[file_id]

    # 回退：遍历目录（兼容旧代码路径）
    for p in state.upload_dir.iterdir():
        if p.name.startswith(file_id) and p.is_file():
            state._file_index[file_id] = p  # 缓存以加速后续查询
            return p
    return None


def _get_media_info(file_path: Path, is_video: bool) -> Dict[str, Any]:
    """获取媒体文件基本信息。"""
    info = {"type": "video" if is_video else "image"}

    if is_video:
        cap = cv2.VideoCapture(str(file_path))
        info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["duration_seconds"] = round(info["frame_count"] / info["fps"], 2) if info["fps"] > 0 else 0
        cap.release()
    else:
        img = cv2.imread(str(file_path))
        if img is not None:
            info["width"] = img.shape[1]
            info["height"] = img.shape[0]

    info["size_mb"] = round(file_path.stat().st_size / (1024 * 1024), 2)
    return info


async def _run_processing_task(task_id: str, request: ProcessRequest):
    """后台执行处理任务的异步包装。"""
    try:
        state.processing_tasks[task_id]["status"] = "processing"

        # 在线程池中执行 CPU/GPU 密集型任务
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _process_sync(task_id, request))

        state.processing_tasks[task_id]["status"] = "completed"
        state.processing_tasks[task_id]["result"] = result
        state.processing_tasks[task_id]["progress"] = 100

    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {e}", exc_info=True)
        state.processing_tasks[task_id]["status"] = "failed"
        state.processing_tasks[task_id]["error"] = str(e)


def _process_sync(task_id: str, request: ProcessRequest) -> Dict[str, Any]:
    """
    同步执行完整的检测+修复流水线（#18 修复：替换 TODO 占位为真实实现）。

    集成 TwoPassProcessor 完整流程：
      1. 解析输入文件路径
      2. 根据请求配置创建检测器和处理器
      3. 执行两轮处理（Pass1 检测 → Pass2 修复）
      4. 返回输出文件信息
    """
    import time
    start_time = time.time()

    # --- 文件定位 ---
    input_name = request.input_filename
    file_path = _find_uploaded_file(input_name)
    if not file_path:
        stem = Path(input_name).stem
        file_path = _find_uploaded_file(stem)
    if not file_path and "_" in input_name:
        file_path = _find_uploaded_file(input_name.split("_")[0])

    if not file_path:
        raise FileNotFoundError(f"输入文件不存在: {input_name}")

    output_path = state.output_dir / f"{task_id}_{file_path.name}"

    try:
        # --- 创建检测器（使用缓存避免重复加载大模型）---
        from backend.tools.watermark_detect import UnifiedWatermarkDetector

        det_config = request.detection
        detector = UnifiedWatermarkDetector.get_or_create(
            detector_type=det_config.detect