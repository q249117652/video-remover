"""
统一水印检测接口 - 整合 PaddleOCR 和 Florence-2 双检测通道

作为 VSR 原有 subtitle_detect.py 的增强替代/补充，
对外提供统一的检测 API，内部根据配置自动选择检测器：

  - DETECTOR_OCR: 使用原有 PaddleOCR（文本/字幕类水印）
  - DETECTOR_FLORENCE: 使用 Florence-2（logo/图标/视觉水印）
  - DETECTOR_HYBRID: 先 OCR 再 Florence-2，合并去重（推荐）

同时整合人工修正流程，形成完整的检测流水线：

  视频帧 → 自动检测(OCR/Florence/Hybrid) → 人工修正(ROICorrector) → 最终 mask

设计原则：
  - 与 VSR 现有代码完全兼容，通过 config.py 切换
  - 不修改原有 ocr.py / subtitle_detect.py，新增独立模块
  - 输出格式与 inpaint_tools.create_mask 兼容
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2

# VSR 原有 OCR 检测链：SubtitleDetect 类（封装 PaddleOCR）
# 注意：VSR 的 ocr.py 只提供 get_coordinates() 函数，没有 OCRDetector 类
# 我们通过 SubtitleDetect 来调用 OCR 检测能力
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.ocr import get_coordinates
from .florence_detector import (
    Florence2Detector,
    WatermarkTypeClassifier,
    ROICorrector,
    MultiFrameMaskVoter,
)

logger = logging.getLogger(__name__)

# 检测器类型常量
DETECTOR_OCR = "paddleocr"
DETECTOR_FLORENCE = "florence2"
DETECTOR_HYBRID = "hybrid"           # OCR + Florence-2 合并
DETECTOR_MANUAL = "manual"            # 纯手动框选


class UnifiedWatermarkDetector:
    """
    统一水印检测器。

    整合多种检测源，提供一致的检测接口。
    支持单帧检测、视频序列检测、带人工修正的完整流程。
    """

    def __init__(
        self,
        detector_type: str = DETECTOR_HYBRID,
        ocr_config: Optional[Dict[str, Any]] = None,
        florence_config: Optional[Dict[str, Any]] = None,
        enable_manual_correction: bool = False,
        enable_watermark_classification: bool = True,
    ):
        """
        Args:
            detector_type: 检测器类型 ('paddleocr' | 'florence2' | 'hybrid' | 'manual')
            ocr_config: PaddleOCR 检测器配置参数字典
            florence_config: Florence-2 检测器配置参数字典
            enable_manual_correction: 是否启用人工修正步骤
            enable_watermark_classification: 是否启用水印分类（用于推荐修复算法）
        """
 