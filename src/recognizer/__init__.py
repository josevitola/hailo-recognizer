import cv2
from pathlib import Path
from picamera2 import Picamera2
from .detector import HailoFaceDetector

# Resolve absolute path to models/ relative to project root
# __file__    -> src/recognizer/__init__.py
# parents[0]  -> src/recognizer
# parents[1]  -> src
# parents[2]  -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "scrfd_2.5g_h8l.hef"


def main():
    # Initialize Hailo detector
    detector = HailoFaceDetector(hef_path=MODEL_PATH, confidence_threshold=0.5)

    # Initialize Picamera2 stream
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    )
    picam2.start()

    print("Starting face detection stream. Press 'q' to exit.")

    try:
        while True:
            # 1. Grab frame from camera (RGB)
            frame_rgb = picam2.capture_array()

            # 2. Run NPU inference
            detections = detector.detect(frame_rgb)

            # 3. Convert frame to BGR for display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 4. Draw bounding boxes and confidence score
            for x1, y1, x2, y2, score in detections:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    f"Face {score:.2f}",
                    (x1, max(10, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Raspberry Pi 5 + Hailo Face Detection", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
