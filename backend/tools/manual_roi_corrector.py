"""
手动框选/修正交互模块 - 从 lxulxu/WatermarkRemover 移植并适配

提供自动检测结果的人工修正能力：
  - 预览自动检测框选结果
  - 拖动调整现有边界框
  - 手动添加遗漏的水印区域
  - 删除误检区域
  - 多帧 mask 投票合并（参考 lxulxu 的多帧采样策略）

与 Florence-2 自动检测配合，形成"自动检测 → 人工修正 → 精确修复"的完整流程。

Bug 修复记录：
  v1.1: 修复 #4 — 补全截断文件，实现全部鼠标交互方法
        新增   — 键盘快捷键 R(重置)/A(清除) 支持
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ROICorrector:
    """
    手动 ROI（感兴趣区域）修正器。

    工作流程：
      1. 接收自动检测器的初始检测结果
      2. 在 GUI 中叠加显示检测框
      3. 用户通过鼠标交互进行修正：
         - 点击已有框可拖动调整大小和位置
         - 右键删除误检框
         - 双击空白区域添加新框（或左键拖动）
         - 按 Enter/Space 确认，按 Esc 取消
      4. 输出修正后的坐标列表

    同时支持纯手动模式（无初始检测结果，完全由用户框选），
    以及多帧投票模式（从多个采样帧生成一致的 mask）。
    """

    # UI 配色
    COLOR_DETECTED = (0, 255, 0)       # 绿色: 自动检测结果
    COLOR_ADDED = (255, 165, 0)        # 橙色: 用户新增的区域
    COLOR_DELETED = (0, 0, 255)        # 红色: 待删除标记
    COLOR_CONFIRMED = (0, 255, 255)    # 黄色: 已确认的最终框
    BOX_THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6

    def __init__(
        self,
        window_name: str = "Watermark ROI Correction",
        show_instructions: bool = True,
    ):
        self.window_name = window_name
        self.show_instructions = show_instructions

        # 内部状态
        self._image = None               # 当前显示的图像副本
        self._original_image = None      # 原始图像（用于重绘）
        self._rois: List[List[int]] = []  # 当前 ROI 列表 [x1, y1, x2, y2]
        self._selected_idx = -1          # 当前选中的 ROI 索引
        self._drag_start = None          # 拖动起始点 (x, y)
        self._drag_mode = None           # 'move' | 'resize_tl' | 'resize_tr' | ...
        self._confirmed = False
        self._cancelled = False
        self._new_roi_start = None      # 新建 ROI 的起点

        # 回调钩子（可选，用于 GUI 集成）
        self._on_roi_added_callback: Optional[Callable] = None
        self._on_roi_removed_callback: Optional[Callable] = None
        self._on_roi_modified_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        on_added=None,
        on_removed=None,
        on_modified=None,
    ):
        """设置 ROI 变更回调函数。"""
        self._on_roi_added_callback = on_added
        self._on_roi_removed_callback = on_removed
        self._on_roi_modified_callback = on_modified

    def correct(
        self,
        image: np.ndarray,
        initial_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        启动人工修正交互。

        Args:
            image: BGR 格式原始图像
            initial_detections: 自动检测器的初始检测结果列表，
                                每个 dict 含 "bbox": [x1,y1,x2,y2]
                                为 None 时进入纯手动模式

        Returns:
            (final_image, corrected_detections) 元组：
              - final_image: 叠加了最终确认框的可视化图像
              - corrected_detections: 修正后的检测结果列表
        """
        if image is None or image.size == 0:
            logger.warning("correct() 收到空图像")
            return (image or np.zeros((100, 100, 3), dtype=np.uint8)), []

        self._original_image = image.copy()
        self._image = image.copy()
        self._confirmed = False
        self._cancelled = False
        self._selected_idx = -1
        self._rois = []

        # 将初始检测结果转为内部 ROI 格式
        if initial_detections:
            for det in initial_detections:
                bbox = det.get("bbox", [])
                if len(bbox) >= 4:
                    self._rois.append([int(v) for v in bbox[:4]])

        # 创建窗口并注册鼠标回调
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
        except cv2.error as e:
            logger.error(f"无法创建 OpenCV 窗口（可能无显示器）: {e}")
            # 无头模式：直接返回初始结果
            corrected = [
                {"bbox": roi, "label": "watermark", "confidence": 1.0, "source": "auto"}
                for roi in self._rois
            ]
            return image.copy(), corrected

        # 绘制初始状态
        self._redraw()

        # 主循环
        logger.info("启动人工修正交互界面")
        while not self._confirmed and not self._cancelled:
            key = cv2.waitKey(30) & 0xFF

            if key == 13 or key == 32:  # Enter 或 Space → 确认
                self._confirmed = True
                logger.info(f"用户确认: 共 {len(self._rois)} 个水印区域")
            elif key == 27:  # Esc → 取消
                self._cancelled = True
                logger.info("用户取消修正")
            elif key == ord('r') or key == ord('R'):  # 重置到初始状态
                self._rois = []
                if initial_detections:
                    for det in initial_detections:
                        bbox = det.get("bbox", [])
                        if len(bbox) >= 4:
                            self._rois.append([int(v) for v in bbox[:4]])
                self._redraw()
                logger.info("重置到初始检测结果")
            elif key == ord('a') or key == ord('A'):  # 全部清除
                self._rois = []
                self._redraw()
                logger.info("清除所有区域")

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

        # 构建返回结果
        result_image = self._image.copy()
        corrected = []
        for roi in self._rois:
            x1, y1, x2, y2 = roi
            cv2.rectangle(result_image, (x1, y1), (x2, y2), self.COLOR_CONFIRMED, 3)
            corrected.append({
                "bbox": [x1, y1, x2, y2],
                "label": "watermark",
                "confidence": 1.0,  # 人工确认为 100% 可信
                "source": "manual_correction",
            })

        return result_image, corrected

    def _mouse_callback(self, event, x, y, flags, param):
        """OpenCV 鼠标事件回调。"""
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._handle_left_down(x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                self._handle_mouse_move(x, y, flags)
            elif event == cv2.EVENT_LBUTTONUP:
                self._handle_left_up(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._handle_right_down(x, y)
            elif event == cv2.EVENT_LDBLCLK:
                self._handle_double_click(x, y)
        except Exception as e:
            logger.debug(f"鼠标回调异常: {e}")

    def _handle_left_down(self, x: int, y: int):
        """左键按下：选中已有 ROI 或开始新建。"""
        clicked_idx = self._find_roi_at(x, y)

        if clicked_idx >= 0:
            # 选中已有 ROI
            self._selected_idx = clicked_idx
            self._drag_start = (x, y)
            self._drag_mode = self._get_drag_mode(x, y, self._rois[clicked_idx])
        else:
            # 开始新建 ROI
            self._selected_idx = -1
            self._new_roi_start = (x, y)

    def _handle_mouse_move(self, x: int, y: int, flags):
        """鼠标移动：拖动或绘制新框预览。"""
        if self._drag_start and self._selected_idx >= 0 and flags & cv2.EVENT_FLAG_LBUTTON:
            dx = x - self._drag_start[0]
            dy = y - self._drag_start[1]

            if self._selected_idx >= len(self._rois):
                return
            roi = self._rois[self._selected_idx].copy()

            if self._drag_mode == "move":
                h, w = self._original_image.shape[:2]
                roi[0] = max(0, min(w, roi[0] + dx))
                roi[1] = max(0, min(h, roi[1] + dy))
                roi[2] = max(0, min(w, roi[2] + dx))
                roi[3] = max(0, min(h, roi[3] + dy))
            elif self._drag_mode and self._drag_mode.startswith("resize_"):
                if "tl" in self._drag_mode:
                    roi[0] = min(roi[2] - 10, x)
                    roi[1] = min(roi[3] - 10, y)
                if "tr" in self._drag_mode:
                    roi[2] = max(roi[0] + 10, x)
                    roi[1] = min(roi[3] - 10, y)
                if "bl" in self._drag_mode:
                    roi[0] = min(roi[2] - 10, x)
                    roi[3] = max(roi[1] + 10, y)
                if "br" in self._drag_mode:
                    roi[2] = max(roi[0] + 10, x)
                    roi[3] = max(roi[1] + 10, y)

            self._rois[self._selected_idx] = roi
            self._drag_start = (x, y)
            self._redraw()

        elif self._new_roi_start is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            # 绘制新建 ROI 预览
            self._redraw()
            cv2.rectangle(
                self._image, self._new_roi_start, (x, y),
                self.COLOR_ADDED, 1, cv2.LINE_AA,
            )

    def _handle_left_up(self, x: int, y: int):
        """左键释放：完成拖动或创建新 ROI。"""
        if self._new_roi_start is not None:
            x1, y1 = self._new_roi_start
            x2, y2 = x, y
            # 确保坐标规范化（左上→右下）
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # 过滤太小的框（最小 10x10）
            if abs(x2 - x1) >= 10 and abs(y2 - y1) >= 10:
                new_roi = [x1, y1, x2, y2]
                self._rois.append(new_roi)
                logger.debug(f"用户新增水印区域: {new_roi}")
                if self._on_roi_added_callback:
                    try:
                        self._on_roi_added_callback(new_roi)
                    except Exception as e:
                        logger.debug(f"回调异常: {e}")

            self._new_roi_start = None
            self._redraw()

        self._drag_start = None
        self._drag_mode = None
        self._selected_idx = -1

    def _handle_right_down(self, x: int, y: int):
        """右键点击：删除点击位置的 ROI。"""
        idx = self._find_roi_at(x, y)
        if idx >= 0 and idx < len(self._rois):
            removed = self._rois.pop(idx)
            logger.debug(f"用户删除水印区域: {removed}")
            if self._on_roi_removed_callback:
                try:
                    self._on_roi_removed_callback(removed)
                except Exception as e:
                    logger.debug(f"回调异常: {e}")
            self._redraw()

    def _handle_double_click(self, x: int, y: int):
        """双击：切换 ROI 的选中状态。"""
        idx = self._find_roi_at(x, y)
        if idx >= 0:
            self._selected_idx = idx if self._selected_idx != idx else -1
            self._redraw()

    def _find_roi_at(self, x: int, y: int) -> int:
        """查找包含点 (x,y) 的 ROI 索引，未找到返回 -1。"""
        for i, roi in enumerate(self._rois):
            if len(roi) >= 4:
                x1, y1, x2, y2 = roi[:4]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return i
        return -1

    @staticmethod
    def _get_drag_mode(x: int, y: int, roi: List[int]) -> Optional[str]:
        """根据点击位置判断拖动模式。"""
        if len(roi) < 4:
            return "move"
        x1, y1, x2, y2 = roi[:4]
        threshold = 8  # 角落判定阈值像素

        corners = {
            "resize_tl": (abs(x - x1) < threshold and abs(y - y1) < threshold),
            "resize_tr": (abs(x - x2) < threshold and abs(y - y1) < threshold),
            "resize_bl": (abs(x - x1) < threshold and abs(y - y2) < threshold),
            "resize_br": (abs(x - x2) < threshold and abs(y - y2) < threshold),
        }

        for mode, hit in corners.items():
            if hit:
                return mode

        return "move"

    def _redraw(self):
        """重绘当前状态。"""
        if self._original_image is None:
            return
        self._image = self._original_image.copy()

        # 绘制操作提示
        if self.show_instructions:
            instructions = [
                "[Enter/Space] Confirm  [Esc] Cancel  [R] Reset  [A] Clear All",
                f"[Left-drag] Add/Move/Resize  [Right-click] Delete  ({len(self._rois)} regions)",
            ]
            for i, line in enumerate(instructions):
                cv2.putText(
                    self._image, line, (10, 20 + i * 20),
                    self.FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    self._image, line, (10, 20 + i * 20),
                    self.FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
                )

        # 绘制所有 ROI
        for i, roi in enumerate(self._rois):
            if len(roi) < 4:
                continue
            x1, y1, x2, y2 = roi[:4]
            color = self.COLOR_CONFIRMED if i == self._selected_idx else self.COLOR_DETECTED
            thickness = self.BOX_THICKNESS + 1 if i == self._selected_idx else self.BOX_THICKNESS

            cv2.rectangle(self._image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # 编号标签
            label = f"#{i+1}"
            (tw, th), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, 1)
            label_y1 = max(0, y1 - th - 6)
            cv2.rectangle(
                self._image, (x1, label_y1), (x1 + tw + 4, label_y1 + th + 6),
                color, -1,
            )
            cv2.putText(
                self._image, label, (x1 + 2, label_y1 + th + 2),
                self.FONT, self.FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA,
            )

        try:
            cv2.imshow(self.window_name, self._image)
        except cv2.error:
            pass


class MultiFrameMaskVoter:
    """
    多帧 mask 投票合并器（移植自 lxulxu WatermarkRemover）。

    从视频中采样 N 帧，分别检测/框选水印区域，
    通过投票机制生成更鲁棒的统一 mask。
    """

    def __init__(
        self,
        sample_count: int = 5,
        dilation_kernel: int = 15,
        dilation_iterations: int = 2,
        blur_kernel_size: int = 21,
        vote_threshold_ratio: float = 0.4,
    ):
        self.sample_count = sample_count
        self.dilation_kernel = dilation_kernel
        self.dilation_iterations = dilation_iterations
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.vote_threshold_ratio = vote_threshold_ratio

    def vote(
        self,
        frame_masks: List[np.ndarray],
        image_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """对多个帧的 mask 进行投票合并。"""
        if not frame_masks:
            h, w = image_shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        h, w = frame_masks[0].shape[:2]

        aligned_masks = []
        for mask in frame_masks:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            aligned_masks.append(mask.astype(np.float32))

        vote_map = sum(aligned_masks) / 255.0

        vote_map_uint8 = (vote_map * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(
            vote_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        if binary_mask.sum() == 0 or binary_mask.sum() == h * w * 255:
            threshold_val = int(255 * self.vote_threshold_ratio)
            _, binary_mask = cv2.threshold(vote_map_uint8, threshold_val, 255, cv2.THRESH_BINARY)

        if self.dilation_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_kernel, self.dilation_kernel),
            )
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=self.dilation_iterations)

        if self.blur_kernel_size > 0:
            blended = cv2.GaussianBlur(binary_mask, (self.blur_kernel_size, self.blur_kernel_size), 0)
        else:
            blended = binary_mask

        logger.info(
            f"多帧投票完成: {len(frame_masks)} 帧 → "
            f"有效面积占比={binary_mask.sum() / (h * w) * 100:.1f}%"
        )

        return blended.astype(np.uint8)

    @staticmethod
    def rois_to_mask(
        rois: List[List[int]],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """将 ROI 坐标列表转换为二值 mask。"""
        mask = np.zeros(shape, dtype=np.uint8)
        for roi in rois:
            if len(roi) >= 4:
                x1, y1, x2, y2 = [int(v) for v in roi[:4]]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(shape[1], x2)
                y2 = min(shape[0], y2)
                mask[y1:y2, x1:x2] = 255
        return mask


def select_roi_simple(
    image: np.ndarray,
    window_name: str = "Select Watermark Region",
) -> Optional[Tuple[int, int, int, int]]:
    """
    简化的单 ROI 选择（兼容 lxulxu 的 selectROI 模式）。
    """
    print("=" * 60)
    print("请框选水印区域:")
    print("  - 鼠标拖动选择区域")
    print("  - 按 Space 或 Enter 确认")
    print("  - 按 Esc 取消")
    print("=" * 60)

    roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if roi[2] > 0 and roi[3] > 0:
        logger.info(f"用户框选 ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        return roi
    else:
        logger.info("用户取消 ROI 选择")
        return None
