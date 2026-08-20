"""Detailed detector benchmark to isolate NPU inference vs post-processing."""

import time
import numpy as np
from pathlib import Path

from recognizer.config import DETECTOR_MODEL_PATH
from recognizer.detector import HailoFaceDetector


def main():
    print("Loading detector...")
    detector = HailoFaceDetector(hef_path=DETECTOR_MODEL_PATH)

    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    detector.detect(test_frame)

    # Measure individual stages
    infer_times = []
    post_times = []
    total_times = []

    for _ in range(10):
        t0 = time.perf_counter()

        # Stage 1: prepare input + NPU inference
        input_data = detector._prepare_input(test_frame)
        t1 = time.perf_counter()
        raw_results = detector._infer(input_data)
        t2 = time.perf_counter()

        # Stage 2: post-processing (sigmoid, decode, NMS)
        # Replicate the post-processing logic to time it separately
        outputs_by_stride = {}
        for name, tensor in raw_results.items():
            _, h, w, c = tensor.shape
            stride = detector.input_height // h
            if stride not in outputs_by_stride:
                outputs_by_stride[stride] = {}
            if c <= 2:
                outputs_by_stride[stride]["score"] = tensor
            elif c == 8:
                outputs_by_stride[stride]["bbox"] = tensor

        all_boxes = []
        all_scores = []
        for stride in detector.strides:
            if stride not in outputs_by_stride:
                continue
            stride_data = outputs_by_stride[stride]
            if "score" not in stride_data or "bbox" not in stride_data:
                continue
            raw_scores = stride_data["score"].flatten()
            sig_scores = detector._sigmoid(raw_scores)
            bbox_outputs = stride_data["bbox"]
            anchors = detector.anchors_by_stride[stride]
            boxes_sig, scores_sig = detector._decode_boxes(anchors, bbox_outputs, stride, sig_scores)
            boxes_raw, scores_raw = detector._decode_boxes(anchors, bbox_outputs, stride, raw_scores)
            if len(boxes_sig) > 0:
                all_boxes.append(boxes_sig)
                all_scores.append(scores_sig)
            elif len(boxes_raw) > 0:
                all_boxes.append(boxes_raw)
                all_scores.append(scores_raw)

        t3 = time.perf_counter()

        infer_times.append((t2 - t1) * 1000)
        post_times.append((t3 - t2) * 1000)
        total_times.append((t3 - t0) * 1000)

    print(f"\n  _prepare_input + _infer: avg={np.mean(infer_times):.1f}ms")
    print(f"  post-processing:         avg={np.mean(post_times):.1f}ms")
    print(f"  total (no NMS):          avg={np.mean(total_times):.1f}ms")

    # Now measure full detect() including NMS
    full_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        detector.detect(test_frame)
        full_times.append((time.perf_counter() - t0) * 1000)
    print(f"  full detect() incl NMS:  avg={np.mean(full_times):.1f}ms")

    # Count candidates to understand NMS load
    input_data = detector._prepare_input(test_frame)
    raw_results = detector._infer(input_data)
    total_candidates = 0
    for name, tensor in raw_results.items():
        _, h, w, c = tensor.shape
        stride = detector.input_height // h
        if c <= 2:
            scores = tensor.flatten()
            sig = detector._sigmoid(scores)
            n_sig = int(np.sum(sig >= detector.confidence_threshold))
            n_raw = int(np.sum(scores >= detector.confidence_threshold))
            print(f"  stride={stride}: sigmoid candidates={n_sig}, raw candidates={n_raw}")
            total_candidates += n_sig

    print(f"\n  Total sigmoid candidates: {total_candidates}")

    detector.close()
    print("Done.")


if __name__ == "__main__":
    main()