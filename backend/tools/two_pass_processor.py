"""
两轮视频处理管线 - 从 D-Ogi/WatermarkRemover-AI 移植并适配到 VSR 架构

解决的核心问题：
  - 逐帧运行 Florence-2 检测太慢
  - 固定水印不需要每帧都检测
  - 渐变水印需要时间轴 mask 扩展

两轮策略（Two-Pass Processing）：

  Pass 1 - 稀疏检测阶段：
    1. 每 N 帧采样一次（detection_skip）
    2. 对采样帧运行 Florence-2/PaddleOCR 检测
    3. 收集所有检测到的水印位置和时间戳
    4. 可选：用户在关键帧上修正检测结果
    5. 沿时间轴扩展 mask（fade-in/fade-out）
    6. 通过插值生成所有帧的 mask

  Pass 2 - 逐帧修复阶段：
    1. 使用 Pass 1 生成的完整 mask 序列
    2. 逐帧调用 VSR 的修复后端（LaMA/STTN/ProPainter）
    3. FFmpeg 合成最终视频（保留音频）

优势：
  - Pass 1 只需对 ~1/skip 的帧跑检测，速度提升 skip 倍
  - Pass 2 是纯 mask+图像处理，无需再加载检测模型
  - 两阶段解耦，可分别优化和并行化
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
import time
import tempfile
import subprocess

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class TwoPassConfig:
    """两轮处理配置。"""
    # Pass 1: 检测配置
    detection_skip: int = 5              # 检测间隔帧数
    fade_in_seconds: float = 0.0         # 向前扩展秒数
    fade_out_seconds: float = 0.0        # 向后扩展秒数
    max_bbox_percent: float = 10.0       # 最大检测框面积百分比
    enable_correction: bool = True       # 是否启用人工修正
    correction_frames: List[int] = field(default_factory=lambda: [0])  # 需要修正的帧号

    # Pass 2: 修复配置
    inpaint_mode: str = "sttn-auto"      # 修复算法
    batch_size: int = 10                 # 批量处理帧数
    preserve_audio: bool = True          # 是否保留音频

    # 输出配置
    output_format: str = "mp4"           # 输出格式
    output_fps: Optional[float] = None   # 输出帧率（None=跟随原视频）
    temp_dir: Optional[str] = None       # 临时目录


class TwoPassProcessor:
    """
    两轮视频处理器。

    整合 Florence-2 检测、人工修正、VSR 多算法修复、FFmpeg 合成的完整流水线。
    """

    def __init__(
        self,
        config: TwoPassConfig,
        detector=None,
        corrector=None,
        inpainter=None,
    ):
        """
        Args:
            config: 两轮处理配置
            detector: UnifiedWatermarkDetector 实例
            corrector: ROICorrector 实例（可选）
            inpainter: VSR 修复器实例（可选，延迟创建）
        """
        self.config = config
        self.detector = detector
        self.corrector = corrector
        self.inpainter = inpainter
        self._temp_dir = None

    @property
    def temp_dir(self) -> Path:
        """获取/创建临时工作目录。"""
        if self._temp_dir is None:
            if self.config.temp_dir:
                self._temp_dir = Path(self.config.temp_dir)
                self._temp_dir.mkdir(parents=True, exist_ok=True)
            else:
                import tempfile as tf
                self._temp_dir = Path(tf.mkdtemp(prefix="twopass_"))
        return self._temp_dir

    def process(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        执行完整的两轮处理流程。

        Args:
            input_path: 输入视频/图片路径
            output_path: 输出路径

        Returns:
            处理结果统计字典
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"{'='*60}")
        logger.info(f"开始两轮处理: {input_path} → {output_path}")
        logger.info(f"配置: skip={self.config.detection_skip}, "
                     f"fade_in={self.config.fade_in_seconds}s, "
                     f"fade_o