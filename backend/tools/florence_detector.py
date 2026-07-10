"""
Florence-2 视觉模型检测器 - 从 D-Ogi/WatermarkRemover-AI 移植并适配

支持任意视觉水印检测（logo、图标、半透明水印、AI生成水印等），
弥补 PaddleOCR 仅能检测文本的不足。

核心功能：
  - identify(): 对单帧图像运行 Florence-2 检测，返回边界框列表
  - get_watermark_mask(): 将检测结果转换为二值 mask
  - detect_only(): 预览模式，仅检测不处理
  - detect_video_frames(): 视频帧序列稀疏检测（detection-skip 策略）
  - expand_mask_temporal(): fade-in/fade-out 时间轴 mask 扩展

依赖：transformers, torch, PIL, numpy, opencv-python

Bug 修复记录：
  v1.1: 修复 #1 — identify() 双重循环 NameError 崩溃
        修复 #11 — 模型名从 base-ft 改为 base（HuggingFace 官方名称）
        新增   — 输入图像空值/格式校验
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class Florence2Detector:
    """
    基于 Microsoft Florence-2 的通用视觉水印检测器。

    使用 <OPEN_VOCABULARY_DETECTION> 任务进行开放词汇目标检测，
    通过 prompt 指定检测目标（默认 "watermark"），可检测任意视觉水印。

    与 VSR 原有 PaddleOCR 检测器互补：
      - PaddleOCR: 文本/字幕类水印，速度快，资源占用低
      - Florence-2: logo/图标/半透明/AI生成水印，泛化能力强

    推荐模型：
      - microsoft/Florence-2-base (更快，~230M 参数)
      - microsoft/Florence-2-large (更准，~770M 参数)
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: Optional[str] = None,
        dtype: str = "float32",
        detection_prompt: str = "watermark",
        max_bbox_percent: float = 10.0,
        confidence_threshold: float = 0.3,
    ):
        self.model_name = model_name
        self.detection_prompt = detection_prompt
        self.max_bbox_percent = max_bbox_percent
        self.confidence_threshold = confidence_threshold

        # 延迟加载模型（首次使用时才初始化）
        self._model = None
        self._processor = None
        self._device = device
        self._dtype_str = dtype

        logger.info(
            f"Florence2Detector 初始化完成 (model={model_name}, "
            f"prompt='{detection_prompt}', max_bbox={max_bbox_percent}%, "
            f"conf_thresh={confidence_threshold})"
        )

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"

    @property
    def dtype(self):
        import torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self._dtype_str, torch.float32)

    def _load_model(self):
        """延迟加载模型和处理器。"""
        if self._model is not None:
            return

        from transformers import (
            AutoProcessor,
            AutoModelForCausalLM,
        )
        import torch

        logger.info(f"正在加载 Florence-2 模型: {self.model_name} ...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        logger.info(f"Florence-2 模型加载完成，设备={self.device}")

    def identify(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        对单帧图像执行水印检测。

        Args:
            image: BGR 格式的 numpy 数组 (H, W, 3)

        Returns:
            检测结果列表，每个元素为字典：
            {
                "bbox": [x1, y1, x2, y2],   # 边界框坐标
                "label": str,                 # 检测标签
                "confidence": float,          # 置信度 (0-1)
            }
        """
        # 边界检查：空图像或无效尺寸
        if image is None or (hasattr(image, 'size') and image.size == 0):
            logger.warning("identify() 收到空图像，返回空结果")
            return []
        if len(image.shape) != 3 or image.shape[2] < 3:
            logger.warning(f"identify() 图像格式异常: shape={image.shape}，期望 (H,W,3)")
            return []

        self._load_model()

        import torch

        # BGR → RGB → PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # 构建输入
        inputs = self._processor(
            text=f"<{self.detection_prompt}>",
            images=pil_image,
            return_tensors="pt",
        ).to(self.device, self.dtype)

        # 推理
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

        # 解析结果
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self._processor.post_process_generation(
            generated_text,
            task="<OPEN_VOCABULARY_DETECTION>",
            image_size=(image.shape[1], image.shape[0]),
        )

        results = []
        img_area = image.shape[0] * image.shape[1]

        # === Bug #1 修复：正确的检测结果提取逻辑 ===
        # Florence-2 不同版本可能使用不同的键名格式
        possible_keys = [
            f"<{self.detection_prompt}>",
            self.detection_prompt,
            "<OPEN_VOCABULARY_DETECTION>",
        ]
        parsed_detections = None
        for key in possible_keys:
            if key in parsed_answer:
                parsed_detections = parsed_answer[key]
                break

        # 如果标准键名都找不到，尝试从任何包含 bbox 的值中提取
        if parsed_detections is None:
            for key, val in parsed_answer.items():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    if "bbox" in val[0]:
                        parsed_detections = val
                        logger.debug(f"Florence-2 输出使用非标准键名: '{key}'")
                        break

        if parsed_detections is None:
            logger.debug(f"Florence-2 未检测到水印（可用键: {list(parsed_answer.keys())})")
            return results

        # === 单层循环遍历检测结果（原代码此处有双重循环 NameError）===
        for det in parsed_detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            label = det.get("label", self.detection_prompt)
            conf = det.get("confidence", 1.0)

            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)

            # 过滤条件 1: 置信度
            if conf < self.confidence_threshold:
                continue

            # 过滤条件 2: 最大面积比例
            if self.max_bbox_percent > 0 and img_area > 0:
                area_percent = (bbox_area / img_area) * 100
                if area_percent > self.max_bbox_percent:
                    logger.debug(
                        f"过滤大尺寸检测框: {label} "
                        f"面积占比={area_percent:.1f}% > {self.max_bbox_percent}%"
                    )
                    continue

            results.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": conf,
            })

        logger.debug(f"Florence-2 检测到 {len(results)} 个水印区域")
        return results

    def get_watermark_mask(
        self,
        image: np.ndarray,
        padding: int = 5,
        dilate_kernel_size: int = 15,
        dilate_iterations: int = 2,
    ) -> np.ndarray:
        """
        将检测结果转换为二值 mask。
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        detections = self.identify(image)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(w, int(x2) + padding)
            y2 = min(h, int(y2) + padding)
            mask[y1:y2, x1:x2] = 255

        if dilate_kernel_size > 0 and len(detections) > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_kernel_size, dilate_kernel_size),
            )
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

        return mask

    def detect_only(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """预览模式：返回检测结果和可视化叠加图。"""
        detections = self.identify(image)
        vis_image = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f'{det["label"]} {conf:.0%}'
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                vis_image, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1
            )
            cv2.putText(
                vis_image, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )

        return vis_image, detections

    def detect_video_frames(
        self,
        frames: List[np.ndarray],
        skip: int = 1,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """视频帧序列稀疏检测。"""
        results = {}
        total = len(frames)

        for i in range(0, total, skip):
            logger.info(f"正在检测帧 {i+1}/{total} (skip={skip})")
            detections = self.identify(frames[i])
            if detections:
                results[i] = detections

        logger.info(
            f"视频检测完成: 共 {total} 帧，检测了 {len(results)} 帧，"
            f"发现 {sum(len(d) for d in results.values())} 个水印区域"
        )
        return results

    @staticmethod
    def expand_mask_temporal(
        frame_masks: Dict[int, np.ndarray],
        total_frames: int,
        fade_in_seconds: float = 0.0,
        fade_out_seconds: float = 0.0,
        fps: float = 30.0,
    ) -> Dict[int, np.ndarray]:
        """沿时间轴扩展 mask（fade-in / fade-out）。"""
        if fade_in_seconds <= 0 and fade_out_seconds <= 0:
            return frame_masks

        fade_in_frames = int(fade_in_seconds * fps)
        fade_out_frames = int(fade_out_seconds * fps)

        detected_indices = sorted(frame_masks.keys())
        if not detected_indices:
            return frame_masks

        expanded = dict(frame_masks)

        for idx in detected_indices:
            mask = frame_masks[idx]

            if fade_in_frames > 0:
                for fi in range(max(0, idx - fade_in_frames), idx):
                    if fi not in expanded:
                        expanded[fi] = mask.copy()

            if fade_out_frames > 0:
                for fi in range(idx + 1, min(total_frames, idx + fade_out_frames + 1)):
                    if fi not in expanded:
                        expanded[fi] = mask.copy()

        logger.info(
            f"Mask 时间轴扩展完成: fade_in={fade_in_frames}帧, "
            f"fade_out={fade_out_frames}帧, "
            f"原始{len(frame_masks)}帧 → 扩展后{len(expanded)}帧"
        )
        return expanded

    @staticmethod
    def feather_mask_edges(
        mask: np.ndarray,
        feather_radius: int = 3,
    ) -> np.ndarray:
        """Mask 边缘羽化处理。"""
        if feather_radius <= 0:
            return mask

        kernel_size = feather_radius * 4 + 1
        feathered = cv2.GaussianBlur(
            mask, (kernel_size, kernel_size), feather_radius
        )
        return feathered.astype(np.uint8)


class WatermarkTypeClassifier:
    """
    水印分类器（借鉴 zuruoke watermark_type 设计思路）。

    根据检测结果的视觉特征对水印分类，用于选择最优修复策略：
      - TEXT: 文本/字幕类 → STTN 或 LaMA
      - LOGO: 图标/logo 类 → LaMA
      - SEMI_TRANSPARENT: 半透明水印 → ProPainter
      - FULL_OVERLAY: 全屏覆盖层 → OpenCV 快速预处理 + LaMA
      - ANIMATED: 动画/移动水印 → ProPainter（时序一致性最好）
    """

    TYPE_TEXT = "text"
    TYPE_LOGO = "logo"
    TYPE_SEMI_TRANSPARENT = "semi_transparent"
    TYPE_FULL_OVERLAY = "full_overlay"
    TYPE_ANIMATED = "animated"

    @classmethod
    def classify(
        cls,
        detections: List[Dict[str, Any]],
        image: np.ndarray,
    ) -> str:
        """根据检测结果和图像特征判断水印类型。"""
        if not detections:
            return cls.TYPE_TEXT

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        areas = []
        aspect_ratios = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            areas.append(area)
            if h > 0:
                aspect_ratios.append(w / h)

        max_area = max(areas) if areas else 0
        avg_ar = sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 1.0
        area_pct = (max_area / img_area) * 100 if img_area > 0 else 0

        if area_pct > 15:
            return cls.TYPE_FULL_OVERLAY
        if len(detections) >= 3 and all(a < img_area * 0.02 for a in areas):
            return cls.TYPE_SEMI_TRANSPARENT
        if area_pct < 2 and avg_ar < 2.0:
            return cls.TYPE_LOGO
        if avg_ar > 3.0:
            return cls.TYPE_TEXT

        return cls.TYPE_TEXT

    @classmethod
    def suggest_inpaint_mode(cls, watermark_type: str) -> str:
        """根据水印类型推荐修复算法。"""
        suggestions = {
            cls.TYPE_TEXT: "sttn-auto",
            cls.TYPE_LOGO: "lama",
            cls.TYPE_SEMI_TRANSPARENT: "propainter",
            cls.TYPE_FULL_OVERLAY: "lama",
            cls.TYPE_ANIMATED: "propainter",
        }
        return suggestions.get(watermark_type, "lama")
