"""
VSR Watermark Remover - Gradio WebUI

基于 Gradio 5.x 构建的现代化 Web 界面，
支持 Docker 一键部署，提供与桌面 GUI 相当的功能覆盖。

功能特性：
  - 视频/图片水印去除（拖拽上传）
  - 自动检测模式切换（PaddleOCR / Florence-2 / Hybrid）
  - 在线检测框可视化与调整
  - 5 种修复算法可选（LaMA / STTN / ProPainter / OpenCV）
  - 实时处理进度显示
  - 处理前后对比预览
  - 批量处理支持
  - 参数高级调节面板

部署方式：
  docker build -f docker/Dockerfile.webui -t vsr-webui .
  docker run -p 7860:7860 --gpus all vsr-webui
  # 访问 http://localhost:7860
"""

import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

# 确保 backend 模块可导入
BACKEND_DIR = Path(__file__).parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import gradio as gr
from gradio import Image as GrImage, Video as GrVideo, File as GrFile

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("webui")


# ============================================================
# 全局状态管理（懒加载模型）
# ============================================================
class AppState:
    """WebUI 全局状态。"""

    def __init__(self):
        self.detector = None
        self.inpaint_models = {}
        self.temp_dir = tempfile.mkdtemp(prefix="vsr_webui_")
        self.current_task_status = "idle"
        self.last_result = None

    def init_detector(self, detector_type: str = "hybrid"):
        """延迟初始化检测器。"""
        if self.detector is not None:
            return self.detector

        from backend.tools.watermark_detect import (
            UnifiedWatermarkDetector,
            DETECTOR_OCR,
            DETECTOR_FLORENCE,
            DETECTOR_HYBRID,
        )

        type_map = {
            "paddleocr": DETECTOR_OCR,
            "florence2": DETECTOR_FLORENCE,
            "hybrid": DETECTOR_HYBRID,
        }
        dt = type_map.get(detector_type, DETECTOR_HYBRID)

        fl_config = {
            "model_name": os.environ.get(
                "FLORENCE_MODEL", "microsoft/Florence-2-base-ft"
            ),
            "device": os.environ.get("DEVICE", None),
            "detection_prompt": os.environ.get("DETECTION_PROMPT", "watermark"),
            "max_bbox_percent": float(os.environ.get("MAX_BBOX_PERCENT", "10")),
            "confidence_threshold": float(os.environ.get("CONF_THRESHOLD", "0.3")),
        }

        self.detector = UnifiedWatermarkDetector(
            detector_type=dt,
            florence_config=fl_config,
            enable_manual_correction=False,
            enable_watermark_classification=True,
        )
        logger.info(f"检测器初始化完成: {detector_type}")
        return self.detector

    def get_inpaint_model(self, mode: str):
        """获取指定模式的修复模型（延迟加载）。"""
        if mode in self.inpaint_models:
            return self.inpaint_models[mode]

        try:
            from backend.constant import InpaintMode
            from backend.inpaint.lama_inpaint import LaMaInpaint
            from backend.inpaint.sttn_auto_inpaint import SttnAutoInpaint
            from backend.inpaint.propainter_inpaint import ProPainterInpaint
            from backend.inpaint.opencv_inpaint import OpenCVInpaint

            mode_key_map = {
                "lama": InpaintMode.LAMA,
                "sttn-auto": InpaintMode.STTN_AUTO,
                "propainter": InpaintMode.PROPAINTER,
                "opencv": InpaintMode.OPENCV,
                "sttn-det": InpaintMode.STTN_DET,
            }
            mode_enum = mode_key_map.get(mode, InpaintMode.LAMA)

            if mode_enum == InpaintMode.LAMA:
                model = LaMaInpaint()
            elif mode_enum == InpaintMode.STTN_AUTO:
                model = SttnAutoInpaint()
            elif mode_enum == InpaintMode.PROPAINTER:
                model = ProPainterInpaint()
            elif mode_enum == InpaintMode.OPENCV:
                model = OpenCVInpaint()
            else:
                from backend.inpaint.sttn_det_inpaint import SttnDetInpaint
                model = SttnDetInpaint()

            self.inpaint_models[mode] = model
            logger.info(f"修复模型加载完成: {mode}")
            return model

        except Exception as e:
            logger.error(f"加载修复模型 [{mode}] 失败: {e}", exc_info=True)
            from backend.inpaint.opencv_inpaint import OpenCVInpaint
            fallback = OpenCVInpaint()
            self.inpaint_models[mode] = fallback
            return fallback


# 全局状态实例
state = AppState()


# ============================================================
# 核心处理函数
# ============================================================

def process_image_webui(
    input_image: np.ndarray,
    detector_type: str,
    inpaint_mode: str,
    padding: int,
    dilate_kernel: int,
    feather_radius: int,
    progress: gr.Progress = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """图片水印去除处理（WebUI 入口）。"""
    if input_image is None:
        return None, {"status": "error", "error": "请先上传图片"}

    metadata = {"status": "processing"}

    try:
        if progress:
            progress(0.1, desc="初始化检测器...")

        detector = state.init_detector(detector_type)
        inpaint_model = state.get_inpaint_model(inpaint_mode)

        if progress:
            progress(0.3, desc="正在检测水印...")

        # 保存临时输入文件
        input_path = os.path.join(state.temp_dir, "input.png")
        output_path = os.path.join(state.temp_dir, "output.png")
        cv2.imwrite(input_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

        # 执行处理
        from backend.tools.two_pass_processor import process_single_image
        result = process_single_image(
            input_path=input_path,
            output_path=output_path,
            detector=detector,
            inpaint_model=inpaint_model,
            transparent=False,
            padding=padding,
            dilate_kernel=dilate_kernel,
            feather_radius=feather_radius,
        )

        if progress:
            progress(0.9, desc="完成!")

        output_bgr = cv2.imread(output_path)
        if output_bgr is not None:
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        else:
            output_rgb = input_image

        metadata.update(result)
        state.last_result = metadata
        return output_rgb, metadata

    except Exception as e:
        logger.error(f"图片处理异常: {e}\n{traceback.format_exc()}")
        metadata = {"status": "error", "error": str(e)}
        return input_image, metadata


def process_video_webui(
    input_video_path: str,
    detector_type: str,
    inpaint_mode: str,
    detection_skip: int,
    fade_in: float,
    fade_out: float,
    max_bbox_pct: float,
    progress: gr.Progress = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """视频水印去除处理（WebUI 入口）。"""
    if not input_video_path:
        return None, None, {"status": "error", "error": "请先上传视频"}

    metadata = {"status": "processing"}

    try:
        if progress:
            progress(0.05, desc="初始化...")

        detector = state.init_detector(detector_type)
        inpaint_model = state.get_inpaint_model(inpaint_mode)

        input_path = input_video_path
        output_path = os.path.join(state.temp_dir, "output.mp4")

        if progress:
            progress(0.1, desc="开始两轮处理...")

        from backend.tools.two_pass_processor import TwoPassVideoProcessor

        processor = TwoPassVideoProcessor(
            detector=detector,
            inpaint_model=inpaint_model,
            detection_skip=detection_skip,
            fade_in_seconds=fade_in,
            fade_out_seconds=fade_out,
            max_bbox_percent=max_bbox_pct,
            preserve_audio=True,
        )

        def on_p1(cur, total):
            if progress:
                progress(0.1 + 0.3 * cur / total, desc=f"Pass1 检测中 ({cur}/{total})...")

        def on_p2(cur, total):
            if progress:
                progress(0.4 + 0.5 * cur / total, desc=f"Pass2 修复中 ({cur}/{total})...")

        processor.set_callbacks(
            on_pass1_progress=on_p1,
            on_pass2_progress=on_p2,
        )

        result = processor.process(
            input_path=input_path,
            output_path=output_path,
            correction_enabled=False,
        )

        if progress:
            progress(0.95, desc="完成!")

        metadata.update(result)
        state.last_result = metadata
        info_text = _format_result_info(result)
        return output_path, info_text, metadata

    except Exception as e:
        logger.error(f"视频处理异常: {e}\n{traceback.format_exc()}")
        metadata = {"status": "error", "error": str(e)}
        info_text = f"❌ 处理失败: {str(e)}"
        return None, info_text, metadata


def detect_and_preview(
    input_image: np.ndarray,
    detector_type: str,
    max_bbox_pct: float,
    conf_threshold: float,
) -> Tuple[np.ndarray, str]:
    """仅检测不修复，用于预览检测结果。"""
    if input_image is None:
        return None, "请先上传图片"

    try:
        detector = state.init_detector(detector_type)

        if hasattr(detector, '_florence_detector') and detector._florence_detector:
            detector._florence_detector.max_bbox_percent = max_bbox_pct
            detector._florence_detector.confidence_threshold = conf_threshold

        rgb = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        annotated, detections, wm_type = detector.detect_frame_with_correction(rgb)

        vis_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        info_lines = [
            f"🔍 检测模式: {detector_type}",
            f"📍 发现水印区域: {len(detections)} 个",
        ]
        if wm_type:
            info_lines.append(f"🏷️ 水印类型: {wm_type}")

        for i, det in enumerate(detections):
            bbox = det["bbox"]
            source = det.get("source", "?")
            conf = det.get("confidence", 0)
            info_lines.append(
                f"  #{i+1}: [{bbox[0]},{bbox[1]}→{bbox[2]},{bbox[3]}]"
                f" conf={conf:.0%} src={source}"
            )

        return vis_rgb, "\n".join(info_lines)

    except Exception as e:
        logger.error(f"检测预览异常: {e}")
        return input_image, f"❌ 检测失败: {str(e)}"


def _format_result_info(result: Dict[str, Any]) -> str:
    """格式化处理结果为可读文本。"""
    lines = ["📊 **处理结果**", "---"]
    status_emoji = {"success": "✅", "error": "❌", "no_watermark_detected": "ℹ️"}
    emoji = status_emoji.get(result.get("status"), "❓")
    lines.append(f"状态: **{emoji} {result.get('status', 'unknown')}**")
    if result.get("error"):
        lines.append(f"错误: {result['error']}")
    lines.append(f"总帧数: {result.get('total_frames', 'N/A')}")
    lines.append(f"检测帧数: {result.get('detected_frames', 'N/A')}")
    lines.append(f"检测数量: {result.get('detection_count', 'N/A')}")
    p1 = result.get('pass1_time', 0)
    p2 = result.get('pass2_time', 0)
    lines.append(f"⏱️ Pass1: {p1:.1f}s | Pass2: {p2:.1f}s | 总计: {p1+p2:.1f}s")
    if result.get('watermark_type'):
        lines.append(f"水印类型: {result['watermark_type']}")
    if result.get('suggested_mode'):
        lines.append(f"推荐算法: {result['suggested_mode']}")
    return "\n".join(lines)


# ============================================================
# Gradio 界面构建
# ============================================================

CUSTOM_CSS = """
.vsr-container { max-width: 1200px; margin: 0 auto; }
.result-info { font-family: monospace; white-space: pre-wrap; }
"""

TITLE = """
<div style="text-align: center;">
  <h1>🎬 VSR Watermark Remover <small style='font-size:0.5em'>WebUI</small></h1>
  <p>AI-Powered Video & Image Watermark Removal — Florence-2 + LaMA + STTN + ProPainter</p>
</div>
"""

DESCRIPTION = """
### 🚀 功能说明
- **自动检测**: PaddleOCR（文本）+ Florence-2（视觉水印）双通道
- **智能分类**: 自动识别水印类型并推荐最优修复算法
- **多种算法**: LaMA / STTN / ProPainter / OpenCV 四种后端
- **两轮处理**: 稀疏检测 → 时间轴扩展 → 逐帧修复
- **边缘羽化**: 消除修复痕迹，输出更自然
"""


def create_ui():
    """构建并返回 Gradio Blocks 界面。"""

    with gr.Blocks(css=CUSTOM_CSS, title="VSR Watermark Remover") as demo:

        gr.HTML(TITLE)

        # ========== Tab 1: 图片处理 ==========
        with gr.TabItem("🖼️ 图片处理", id="image_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = GrImage(
                        label="上传图片", type="numpy", height=400,
                    )
                    with gr.Accordion("⚙️ 检测参数", open=False):
                        img_det_type = gr.Dropdown(
                            choices=["hybrid", "paddleocr", "florence2"],
                            value="hybrid",
                            label="检测模式",
                            info="hybrid=OCR+Florence合并(推荐)",
                        )
                        img_max_bbox = gr.Slider(
                            1, 50, value=10, step=1,
                            label="最大检测框面积 (%)",
                        )
                        img_conf = gr.Slider(
                            0.0, 1.0, value=0.3, step=0.05,
                            label="置信度阈值",
                        )
                    with gr.Accordion("🔧 修复参数", open=False):
                        img_inpaint_mode = gr.Dropdown(
                            choices=["lama", "sttn-auto", "propainter", "opencv"],
                            value="lama",
                            label="修复算法",
                            info="lama=通用最优, sttn-auto=真人视频",
                        )
                        img_padding = gr.Slider(0, 30, value=5, step=1, label="检测框扩展 (px)")
                        img_dilate = gr.Slider(0, 50, value=15, step=1, label="膨胀核大小")
                        img_feather = gr.Slider(0, 15, value=3, step=1, label="羽化半径")

                    with gr.Row():
                        btn_detect = gr.Button("🔍 仅检测预览", variant="secondary")
                        btn_process_img = gr.Button("🚀 开始处理", variant="primary")

                with gr.Column(scale=1):
                    img_output = GrImage(label="处理结果", type="numpy", height=400)
                    img_info = gr.Textbox(
                        label="检测/处理信息", lines=8,
                        interactive=False, show_copy_button=True,
                    )

        # ========== Tab 2: 视频处理 ==========
        with gr.TabItem("🎬 视频处理", id="video_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = GrVideo(label="上传视频", height=300)
                    with gr.Accordion("⚙️ 检测参数", open=False):
                        vid_det_type = gr.Dropdown(
                            choices=["hybrid", "paddleocr", "florence2"],
                            value="hybrid", label="检测模式",
                        )
                        vid_skip = gr.Slider(
                            1, 30, value=3, step=1,
                            label="检测间隔 (帧)",
                            info="值越大速度越快但精度降低",
                        )
                        vid_fade_in = gr.Slider(
                            0, 5.0, value=0.5, step=0.1, label="Fade-in 扩展 (秒)",
                        )
                        vid_fade_out = gr.Slider(
                            0, 5.0, value=0.5, step=0.1, label="Fade-out 扩展 (秒)",
                        )
                        vid_max_bbox = gr.Slider(
                            1, 50, value=10, step=1, label="最大检测框面积 (%)",
                        )
                    with gr.Accordion("🔧 修复参数", open=False):
                        vid_inpaint_mode = gr.Dropdown(
                            choices=["lama", "sttn-auto", "propainter", "opencv"],
                            value="lama", label="修复算法",
                        )

                    btn_process_vid = gr.Button("🚀 开始处理视频", variant="primary")

                with gr.Column(scale=1):
                    vid_output = GrVideo(label="处理结果", height=300)
                    vid_info = gr.Textbox(
                        label="处理信息", lines=8,
                        interactive=False, show_copy_button=True,
                    )
                    dl_btn = gr.File(label="下载结果", visible=False)

        # ========== Tab 3: 高级选项 ==========
        with gr.TabItem("⚙️ 高级选项 & 使用指南", id="advanced_tab"):
            gr.Markdown("""
            ### 📋 算法选择指南

            | 场景 | 推荐算法 | 说明 |
            |------|---------|------|
            | 动画/插画视频 | **LaMA** | 单帧效果最好，速度快 |
            | 真人视频 | **STTN-Auto** | 利用时序信息，效果自然 |
            | 剧烈运动画面 | **ProPainter** | 效果最好，显存消耗大 |
            | 快速预览/低配设备 | **OpenCV** | 速度最快，质量一般 |

            ### 🔍 检测模式说明

            | 模式 | 适用水印类型 | 速度 |
            |------|------------|------|
            | Hybrid (推荐) | 文本 + Logo + 图标 | 中等 |
            | PaddleOCR | 字幕/文字水印 | 快 |
            | Florence-2 | Logo/图标/AI水印 | 较慢 |

            ### 💡 Tips
            - 固定位置水印：增大 `检测间隔` 到 10-15 可大幅提速
            - 渐变水印：适当增加 `Fade-in/out` 到 1-2 秒
            - 小 logo 水印：减小 `置信度阈值` 到 0.1-0.2
            - 多个分散水印：Hybrid 模式自动合并去重
            """)

        # ========== 事件绑定 ==========
        btn_detect.click(
            fn=detect_and_preview,
            inputs=[img_input, img_det_type, img_max_bbox, img_conf],
            outputs=[img_output, img_info],
        )

        btn_process_img.click(
            fn=process_image_webui,
            inputs=[
                img_input, img_det_type, img_inpaint_mode,
                img_padding, img_dilate, img_feather,
            ],
            outputs=[img_output, img_info],
        )

        btn_process_vid.click(
            fn=process_video_webui,
            inputs=[
                vid_input, vid_det_type, vid_inpaint_mode,
                vid_skip, vid_fade_in, vid_fade_out, vid_max_bbox,
            ],
            outputs=[vid_output, vid_info, dl_btn],
        )

    return demo


# ============================================================
# 启动入口
# ============================================================

def main():
    """启动 WebUI 服务。"""
    import argparse
    parser = argparse.ArgumentParser(description="VSR Watermark Remover WebUI")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="创建公共分享链接")
    parser.add_argument("--temp-dir", default=None, help="临时文件目录")
    args = parser.parse_args()

    if args.temp_dir:
        state.temp_dir = args.temp_dir

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
