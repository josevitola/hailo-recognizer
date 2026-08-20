"""Benchmark HailoRT inference speed for detector and embedder models."""

import time
import numpy as np
from pathlib import Path

from recognizer.config import DETECTOR_MODEL_PATH, EMBEDDER_MODEL_PATH
from recognizer.detector import HailoFaceDetector
from recognizer.embedder import HailoEmbedder


def timeit(label: str, fn, iterations: int = 10) -> float:
    """Run fn `iterations` times and return average ms per call."""
    # Warmup
    fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  {label}: avg={avg:.1f}ms  min={min(times):.1f}ms  max={max(times):.1f}ms")
    return avg


def main():
    print("Loading models...")
    detector = HailoFaceDetector(hef_path=DETECTOR_MODEL_PATH)
    embedder = HailoEmbedder(hef_path=EMBEDDER_MODEL_PATH)

    # Synthetic test image at camera resolution
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    test_crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    print("\n=== Detector (scrfd_2.5g) ===")
    timeit("detect() full", lambda: detector.detect(test_frame), iterations=10)

    print("\n=== Embedder (arcface) ===")
    timeit("extract_embedding()", lambda: embedder.extract_embedding(test_crop), iterations=10)

    print("\n=== Alternating (simulates real pipeline) ===")
    def alternating():
        detector.detect(test_frame)
        embedder.extract_embedding(test_crop)
    timeit("detect + embed", alternating, iterations=10)

    print("\n=== Cleanup ===")
    detector.close()
    embedder.close()
    print("Done.")


if __name__ == "__main__":
    main()