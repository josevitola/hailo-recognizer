import cv2
import numpy as np
from pathlib import Path
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


class HailoFaceDetector:
    def __init__(self, hef_path: str | Path, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.hef = HEF(str(hef_path))
        self.target = VDevice()

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

        # Pre-generate feature strides and anchor grids for SCRFD
        self.strides = [8, 16, 32]
        self.num_anchors_per_stride = 2
        self.anchors = self._generate_anchors()

    def _generate_anchors(self) -> dict[int, np.ndarray]:
        anchors = {}
        for stride in self.strides:
            feat_h = self.input_height // stride
            feat_w = self.input_width // stride
            grid_y, grid_x = np.mgrid[:feat_h, :feat_w]
            grid = np.stack((grid_x, grid_y), axis=-1).astype(np.float32) * stride
            # Duplicate for 2 anchors per stride cell
            anchor_centers = np.repeat(grid[:, :, np.newaxis, :], self.num_anchors_per_stride, axis=2)
            anchors[stride] = anchor_centers.reshape(-1, 2)
        return anchors

    def detect(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        orig_h, orig_w, _ = frame_rgb.shape
        resized_frame = cv2.resize(frame_rgb, (self.input_width, self.input_height))
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
                results = infer_pipeline.infer(input_data)

        boxes, scores = [], []

        # Parse score and bbox tensors across feature strides
        for stride in self.strides:
            # Find output tensors matching current stride resolution
            feat_h, feat_w = self.input_height // stride, self.input_width // stride
            score_tensor = None
            bbox_tensor = None

            for name, tensor in results.items():
                if tensor.shape[1:3] == (feat_h, feat_w):
                    if tensor.shape[-1] in (1, 2):  # Score map
                        score_tensor = tensor[0]
                    elif tensor.shape[-1] in (4, 8):  # Bbox distance map
                        bbox_tensor = tensor[0]

            if score_tensor is None or bbox_tensor is None:
                continue

            # Reshape raw maps into lists of candidate detections
            stride_scores = score_tensor.reshape(-1)
            stride_bboxes = bbox_tensor.reshape(-1, 4) * stride
            stride_anchors = self.anchors[stride]

            mask = stride_scores >= self.confidence_threshold
            if not np.any(mask):
                continue

            valid_scores = stride_scores[mask]
            valid_deltas = stride_bboxes[mask]
            valid_anchors = stride_anchors[mask]

            # Decode distance offsets (left, top, right, bottom) into (x1, y1, x2, y2)
            x1 = valid_anchors[:, 0] - valid_deltas[:, 0]
            y1 = valid_anchors[:, 1] - valid_deltas[:, 1]
            x2 = valid_anchors[:, 0] + valid_deltas[:, 2]
            y2 = valid_anchors[:, 1] + valid_deltas[:, 3]

            boxes.append(np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1))
            scores.append(valid_scores)

        if not boxes:
            return []

        all_boxes = np.vstack(boxes)
        all_scores = np.concatenate(scores)

        # Apply OpenCV NMS to filter overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            bboxes=all_boxes.tolist(),
            scores=all_scores.tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

        detections = []
        if len(indices) > 0:
            scale_x = orig_w / self.input_width
            scale_y = orig_h / self.input_height

            for idx in indices.flatten():
                x, y, w, h = all_boxes[idx]
                score = float(all_scores[idx])

                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)

                detections.append((x1, y1, x2, y2, score))

        return detections