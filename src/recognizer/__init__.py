import argparse
import cv2
from pathlib import Path
from picamera2 import Picamera2

from recognizer.detector import HailoFaceDetector
from recognizer.embedder import HailoEmbedder
from recognizer.matcher import TargetMatcher
from recognizer.config import DETECTOR_MODEL_PATH, EMBEDDER_MODEL_PATH, TARGET_PROFILE_PATH, DETECTOR_THRESHOLD, MATCHER_THRESHOLD
from recognizer.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        prog="preview",
        description="Run the live single-subject face recognition stream.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (detailed detector logs).",
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    logger.info("Loading NPU models and target profile...")
    detector = HailoFaceDetector(hef_path=DETECTOR_MODEL_PATH, confidence_threshold=DETECTOR_THRESHOLD)
    embedder = HailoEmbedder(hef_path=EMBEDDER_MODEL_PATH)
    matcher = TargetMatcher(profile_path=TARGET_PROFILE_PATH, threshold=MATCHER_THRESHOLD)

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    )
    picam2.start()

    logger.info("Live recognition active. Press 'q' to quit.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            h, w, _ = frame_rgb.shape

            # 1. Detect faces in frame
            detections = detector.detect(frame_rgb)

            for x1, y1, x2, y2, det_score in detections:
            # 1. Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue

                # 2. Extract embedding and check match
                embedding = embedder.extract_embedding(crop)
                is_match, similarity = matcher.match(embedding)

                # 3. ONLY DRAW IF MATCHED TO TARGET
                if is_match:
                    label = f"Target ({similarity:.2f})"
                    color = (0, 255, 0)  # Green

                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_rgb,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            cv2.imshow("Raspberry Pi 5 - Target Face Recognition", frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.close()
        embedder.close()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()