import cv2
from pathlib import Path
from picamera2 import Picamera2

from recognizer.detector import HailoFaceDetector
from recognizer.embedder import HailoEmbedder
from recognizer.matcher import TargetMatcher
from recognizer.config import DETECTOR_MODEL_PATH, EMBEDDER_MODEL_PATH, TARGET_PROFILE_PATH

def main():
    print("Loading NPU models and target profile...")
    detector = HailoFaceDetector(hef_path=DETECTOR_MODEL_PATH, confidence_threshold=0.5)
    embedder = HailoEmbedder(hef_path=EMBEDDER_MODEL_PATH)
    matcher = TargetMatcher(profile_path=TARGET_PROFILE_PATH, threshold=0.45)

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    )
    picam2.start()

    print("Live recognition active. Press 'q' to quit.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            h, w, _ = frame_rgb.shape

            # 1. Detect faces in frame
            detections = detector.detect(frame_rgb)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            for x1, y1, x2, y2, det_score in detections:
                # Clamp boundaries
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)

                face_crop = frame_rgb[cy1:cy2, cx1:cx2]
                if face_crop.size == 0:
                    continue

                # 2. Extract 512-D embedding via ArcFace
                live_embedding = embedder.extract_embedding(face_crop)

                # 3. Match against loaded target profile
                is_target, similarity = matcher.match(live_embedding)

                # 4. Render results (Green = Target, Gray = Unknown)
                color = (0, 255, 0) if is_target else (128, 128, 128)
                label = f"Target ({similarity:.2f})" if is_target else f"Unknown ({similarity:.2f})"

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.imshow("Raspberry Pi 5 - Target Face Recognition", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
