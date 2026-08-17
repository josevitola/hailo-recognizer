from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

from recognizer.config import (
    DETECTOR_MODEL_PATH,
    EMBEDDER_MODEL_PATH,
    RAW_TARGET_DIR,
    TARGET_PROFILE_PATH,
)
from recognizer.detector import HailoFaceDetector
from recognizer.embedder import HailoEmbedder


def load_image_rgb(img_path: Path) -> np.ndarray:
    pil_img = Image.open(img_path)
    exif_img = ImageOps.exif_transpose(pil_img)
    arr = np.array(exif_img.convert("RGB"))
    print(f"[DEBUG enroll] Loaded '{img_path.name}': raw shape {pil_img.size} -> exif orientation shape {arr.shape[:2]}")
    return arr


def passes_quality_checks(
    crop: np.ndarray, min_size: int = 80, min_blur_var: float = 100.0
) -> bool:
    h, w, _ = crop.shape
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    print(f"[DEBUG enroll] Quality check crop: {w}x{h} px (Min required: {min_size}x{min_size}) | Blur Variance: {blur_score:.2f} (Min required: {min_blur_var})")

    if h < min_size or w < min_size:
        print("[DEBUG enroll] Quality failed: Resolution too small.")
        return False

    if blur_score < min_blur_var:
        print("[DEBUG enroll] Quality failed: Image too blurry.")
        return False

    return True


def enroll_target() -> None:
    print("Initializing Hailo models for target enrollment...")
    detector = HailoFaceDetector(hef_path=DETECTOR_MODEL_PATH, confidence_threshold=0.3)
    embedder = HailoEmbedder(hef_path=EMBEDDER_MODEL_PATH)

    if not RAW_TARGET_DIR.exists():
        print(f"Error: Target directory '{RAW_TARGET_DIR}' does not exist.")
        return

    image_paths = (
        list(RAW_TARGET_DIR.glob("*.jpg"))
        + list(RAW_TARGET_DIR.glob("*.jpeg"))
        + list(RAW_TARGET_DIR.glob("*.png"))
    )

    if not image_paths:
        print(f"No valid image files found in '{RAW_TARGET_DIR}'.")
        return

    print(f"Processing {len(image_paths)} target images from '{RAW_TARGET_DIR}'...")

    embeddings = []
    valid_count = 0

    for img_path in image_paths:
        print(f"\n==================== Processing {img_path.name} ====================")
        try:
            image_rgb = load_image_rgb(img_path)
        except Exception as e:
            print(f"  [ERROR] Failed to load {img_path.name}: {e}")
            continue

        detections = detector.detect(image_rgb)

        if not detections:
            print(f"  [SKIPPED] {img_path.name}: No face detected by detector.")
            continue

        best_det = max(detections, key=lambda d: d[4])
        x1, y1, x2, y2, score = best_det
        print(f"[DEBUG enroll] Best detection: BBox=[{x1}, {y1}, {x2}, {y2}] | Confidence={score:.4f}")

        h_img, w_img, _ = image_rgb.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        crop = image_rgb[y1:y2, x1:x2]

        if not passes_quality_checks(crop):
            print(f"  [SKIPPED] {img_path.name}: Crop failed quality or resolution checks.")
            continue

        embedding = embedder.extract_embedding(crop)
        embeddings.append(embedding)
        valid_count += 1
        print(f"  [ACCEPTED] {img_path.name} (Confidence: {score:.2f})")

    if not embeddings:
        print("\nEnrollment failed: No image crops passed detection and quality checks.")
        return

    embeddings_matrix = np.vstack(embeddings)
    centroid = np.mean(embeddings_matrix, axis=0)

    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    TARGET_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TARGET_PROFILE_PATH, centroid)

    print(f"\nSuccessfully enrolled target using {valid_count}/{len(image_paths)} images.")
    print(f"Target profile vector saved to: {TARGET_PROFILE_PATH}")


if __name__ == "__main__":
    enroll_target()