from pathlib import Path
import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

from recognizer.device import get_vdevice
from recognizer.logging import get_logger

logger = get_logger(__name__)


class HailoFaceDetector:
    def __init__(
        self,
        hef_path: str | Path,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        target: VDevice | None = None,
    ):
        self.hef = HEF(str(hef_path))
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.target = target or get_vdevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_height = input_vstream_info.shape[0]
        self.input_width = input_vstream_info.shape[1]

        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

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

        resized_frame = cv2.resize(
            image_rgb, (self.input_width, self.input_height)
        )
        input_data = {
            self.hef.get_input_vstream_infos()[0].name: np.expand_dims(
                resized_frame, axis=0
            )
        }

        with InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        ) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                raw_results = infer_pipeline.infer(input_data)

        logger.debug("--- Raw Tensor Output Summary ---")
        outputs_by_stride = {}
        for name, tensor in raw_results.items():
            _, h, w, c = tensor.shape
            stride = self.input_height // h
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

            raw_scores = stride_data["score"].flatten()
            sig_scores = self._sigmoid(raw_scores)

            top5_raw = np.sort(raw_scores)[-5:][::-1]
            top5_sig = np.sort(sig_scores)[-5:][::-1]

            logger.debug("  - Stride %s:", stride)
            logger.debug("      Top 5 Raw Scores:      %s", np.round(top5_raw, 4))
            logger.debug("      Top 5 Sigmoid Scores:  %s", np.round(top5_sig, 4))
            logger.debug("      Confidence Threshold:  %s", self.confidence_threshold)

            bbox_outputs = stride_data["bbox"]
            anchors = self.anchors_by_stride[stride]

            # Try both raw and sigmoid scores for candidates
            boxes_sig, scores_sig = self._decode_boxes(
                anchors, bbox_outputs, stride, sig_scores
            )
            boxes_raw, scores_raw = self._decode_boxes(
                anchors, bbox_outputs, stride, raw_scores
            )

            logger.debug("      Candidates via Sigmoid: %s", len(boxes_sig))
            logger.debug("      Candidates via Raw:     %s", len(boxes_raw))

            # Prefer sigmoid if candidates exist, otherwise try raw
            if len(boxes_sig) > 0:
                all_boxes.append(boxes_sig)
                all_scores.append(scores_sig)
            elif len(boxes_raw) > 0:
                all_boxes.append(boxes_raw)
                all_scores.append(scores_raw)

        if not all_boxes:
            logger.debug("Result: ZERO candidates passed confidence threshold.")
            return []

        cat_boxes = np.vstack(all_boxes)
        cat_scores = np.concatenate(all_scores)

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