import logging
from pathlib import Path
import cv2
import numpy as np
from hailo_platform import VDevice

from recognizer.base import HailoModelBase
from recognizer.logging import get_logger

logger = get_logger(__name__)


class HailoFaceDetector(HailoModelBase):
    def __init__(
        self,
        hef_path: str | Path,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        target: VDevice | None = None,
    ):
        super().__init__(hef_path, target)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.strides = [8, 16, 32]
        self.anchors_by_stride = self._generate_anchors()

    def _generate_anchors(self) -> dict[int, np.ndarray]:
        anchors = {}
        for stride in self.strides:
            grid_h = self.input_height // stride
            grid_w = self.input_width // stride

            grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            anchor_x = (grid_x.flatten() + 0.5) * stride
            anchor_y = (grid_y.flatten() + 0.5) * stride

            centers = np.stack([anchor_x, anchor_y], axis=-1)
            anchors[stride] = np.repeat(centers, 2, axis=0)

        return anchors

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _decode_boxes(
        self,
        anchors: np.ndarray,
        bbox_outputs: np.ndarray,
        stride: int,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return np.empty((0, 4)), np.empty((0,))

        filtered_anchors = anchors[mask]
        filtered_scores = scores[mask]
        deltas = bbox_outputs.reshape(-1, 4)[mask] * stride

        x1 = filtered_anchors[:, 0] - deltas[:, 0]
        y1 = filtered_anchors[:, 1] - deltas[:, 1]
        x2 = filtered_anchors[:, 0] + deltas[:, 2]
        y2 = filtered_anchors[:, 1] + deltas[:, 3]

        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        return boxes, filtered_scores

    def detect(self, image_rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        orig_h, orig_w, _ = image_rgb.shape
        logger.debug("Input Image Shape: %dx%d", orig_w, orig_h)

        input_data = self._prepare_input(image_rgb)

        # Execute on persistent stream pipeline
        raw_results = self._infer(input_data)

        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            logger.debug("--- Raw Tensor Output Summary ---")
        outputs_by_stride = {}
        for name, tensor in raw_results.items():
            _, h, w, c = tensor.shape
            stride = self.input_height // h

            if debug_enabled:
                min_val, max_val, mean_val = tensor.min(), tensor.max(), tensor.mean()
                logger.debug(
                    "  - Key: '%s' | Shape: %s | Stride: %s | Range: [%.4f, %.4f] | Mean: %.4f",
                    name,
                    tensor.shape,
                    stride,
                    min_val,
                    max_val,
                    mean_val,
                )

            if stride not in outputs_by_stride:
                outputs_by_stride[stride] = {}

            if c <= 2:
                outputs_by_stride[stride]["score"] = tensor
            elif c == 8:
                outputs_by_stride[stride]["bbox"] = tensor

        all_boxes = []
        all_scores = []

        logger.debug("--- Stride Decoding Summary ---")
        for stride in self.strides:
            if stride not in outputs_by_stride:
                logger.debug("  - Stride %s: MISSING in output tensors", stride)
                continue

            stride_data = outputs_by_stride[stride]
            if "score" not in stride_data or "bbox" not in stride_data:
                logger.debug("  - Stride %s: Missing score or bbox tensor", stride)
                continue

            sig_scores = self._sigmoid(stride_data["score"].flatten())

            if debug_enabled:
                top5_sig = np.sort(sig_scores)[-5:][::-1]
                logger.debug("  - Stride %s:", stride)
                logger.debug("      Top 5 Sigmoid Scores:  %s", np.round(top5_sig, 4))
                logger.debug("      Confidence Threshold:  %s", self.confidence_threshold)

            bbox_outputs = stride_data["bbox"]
            anchors = self.anchors_by_stride[stride]

            # SCRFD outputs logits; sigmoid is the correct score transform.
            # Only decode once with sigmoid scores (no raw-score fallback).
            boxes, scores = self._decode_boxes(
                anchors, bbox_outputs, stride, sig_scores
            )

            if debug_enabled:
                logger.debug("      Candidates: %s", len(boxes))

            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)

        if not all_boxes:
            logger.debug("Result: ZERO candidates passed confidence threshold.")
            return []

        cat_boxes = np.vstack(all_boxes)
        cat_scores = np.concatenate(all_scores)

        # Limit candidates to top-K by score before NMS.
        # NMS is O(n^2), so processing all ~16K anchors is extremely slow.
        # Keeping only the top 1000 candidates reduces NMS cost dramatically
        # while preserving detection accuracy.
        MAX_CANDIDATES = 1000
        if len(cat_scores) > MAX_CANDIDATES:
            top_indices = np.argsort(cat_scores)[-MAX_CANDIDATES:]
            cat_boxes = cat_boxes[top_indices]
            cat_scores = cat_scores[top_indices]

        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height

        cat_boxes[:, [0, 2]] *= scale_x
        cat_boxes[:, [1, 3]] *= scale_y

        boxes_xywh = cat_boxes.copy()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            cat_scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )

        logger.debug("Total candidates before NMS: %s", len(cat_boxes))
        logger.debug("Total surviving after NMS:  %s", len(indices))

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        results = []
        for idx in indices:
            x1, y1, x2, y2 = cat_boxes[idx].astype(int)
            score = float(cat_scores[idx])
            results.append((x1, y1, x2, y2, score))

        return results