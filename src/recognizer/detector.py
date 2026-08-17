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

        # Use shared VDevice singleton if none passed explicitly
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

            # Shape: (N, 2) where N = grid_h * grid_w (2 anchors per center point in SCRFD)
            centers = np.stack([anchor_x, anchor_y], axis=-1)
            anchors[stride] = np.repeat(centers, 2, axis=0)

        return anchors

    def _decode_boxes(
        self,
        anchors: np.ndarray,
        bbox_outputs: np.ndarray,
        stride: int,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Filter candidate anchor locations by confidence threshold
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

        all_boxes = []
        all_scores = []

        # Parse feature map tensors per stride
        for stride in self.strides:
            score_key = f"scrfd_2.5g_h8l/score_s{stride}"
            bbox_key = f"scrfd_2.5g_h8l/bbox_s{stride}"

            print("Raw output keys:", list(raw_results.keys()))

            if score_key not in raw_results or bbox_key not in raw_results:
                continue

            scores = raw_results[score_key].flatten()
            bbox_outputs = raw_results[bbox_key]
            anchors = self.anchors_by_stride[stride]

            boxes, filtered_scores = self._decode_boxes(
                anchors, bbox_outputs, stride, scores
            )
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(filtered_scores)

        if not all_boxes:
            return []

        cat_boxes = np.vstack(all_boxes)
        cat_scores = np.concatenate(all_scores)

        # Rescale normalized 640x640 coordinates back to original image aspect ratio
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height

        cat_boxes[:, [0, 2]] *= scale_x
        cat_boxes[:, [1, 3]] *= scale_y

        # OpenCV Non-Maximum Suppression (NMS)
        boxes_xywh = cat_boxes.copy()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # Width
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # Height

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            cat_scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        results = []
        for idx in indices:
            x1, y1, x2, y2 = cat_boxes[idx].astype(int)
            score = float(cat_scores[idx])
            results.append((x1, y1, x2, y2, score))

        return results