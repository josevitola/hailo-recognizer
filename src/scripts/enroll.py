import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

from recognizer.config import (
    ACCEPTED_DIR,
    DETECTOR_MODEL_PATH,
    EMBEDDER_MODEL_PATH,
    RAW_TARGET_DIR,
    TARGET_PROFILE_PATH,
)
from recognizer.detector import HailoFaceDetector
from recognizer.embedder import HailoEmbedder
from recognizer.logging import get_logger, setup_logging

logger = get_logger(__name__)


def load_image_rgb(img_path: Path) -> np.ndarray:
    pil_img = Image.open(img_path)
    exif_img = ImageOps.exif_transpose(pil_img)
    arr = np.array(exif_img.convert("RGB"))
    logger.debug(
        "Loaded '%s': raw shape %s -> exif orientation shape %s",
        img_path.name,
        pil_img.size,
        arr.shape[:2],
    )
    return arr


def passes_quality_checks(
    crop: np.ndarray, min_size: int = 80, min_blur_var: float = 100.0
) -> bool:
    h, w, _ = crop.shape
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    logger.debug(
        "Quality check crop: %dx%d px (Min required: %dx%d) | Blur Variance: %.2f (Min required: %s)",
        w,
        h,
        min_size,
        min_size,
        blur_score,
        min_blur_var,
    )

    if h < min_size or w < min_size:
        logger.debug("Quality failed: Resolution too small.")
        return False

    if blur_score < min_blur_var:
        logger.debug("Quality failed: Image too blurry.")
        return False

    return True


def enroll_target() -> None:
    logger.info("Initializing Hailo models for target enrollment...")
    detector = HailoFaceDetector(hef_path=DETECTOR_MODEL_PATH, confidence_threshold=0.3)
    embedder = HailoEmbedder(hef_path=EMBEDDER_MODEL_PATH)

    try:
        if not RAW_TARGET_DIR.exists():
            logger.error("Target directory '%s' does not exist.", RAW_TARGET_DIR)
            return

        supported_extensions = {".jpg", ".jpeg", ".png"}
        image_paths = sorted(
            p
            for p in RAW_TARGET_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in supported_extensions
        )

        if not image_paths:
            logger.error("No valid image files found in '%s'.", RAW_TARGET_DIR)
            return

        logger.info(
            "Processing %d target images from '%s'...", len(image_paths), RAW_TARGET_DIR
        )

        embeddings = []
        valid_count = 0

        for img_path in image_paths:
            logger.info("==================== Processing %s ====================", img_path.name)
            try:
                image_rgb = load_image_rgb(img_path)
            except Exception as e:
                logger.error("Failed to load %s: %s", img_path.name, e)
                continue

            detections = detector.detect(image_rgb)

            if not detections:
                logger.debug("[SKIPPED] %s: No face detected by detector.", img_path.name)
                continue

            best_det = max(detections, key=lambda d: d[4])
            x1, y1, x2, y2, score = best_det
            logger.debug(
                "Best detection: BBox=[%d, %d, %d, %d] | Confidence=%.4f",
                x1,
                y1,
                x2,
                y2,
                score,
            )

            h_img, w_img, _ = image_rgb.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            crop = image_rgb[y1:y2, x1:x2]

            if not passes_quality_checks(crop):
                logger.debug("[SKIPPED] %s: Crop failed quality or resolution checks.", img_path.name)
                continue

            embedding = embedder.extract_embedding(crop)
            embeddings.append(embedding)
            valid_count += 1

            # Save the accepted face crop to data/accepted/
            ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
            accepted_path = ACCEPTED_DIR / img_path.name
            cv2.imwrite(str(accepted_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            logger.info("[ACCEPTED] %s (Confidence: %.2f)", img_path.name, score)
            logger.debug("Saved accepted crop to: %s", accepted_path)

        if not embeddings:
            logger.error("Enrollment failed: No image crops passed detection and quality checks.")
            return

        embeddings_matrix = np.vstack(embeddings)
        centroid = np.mean(embeddings_matrix, axis=0)

        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        TARGET_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(TARGET_PROFILE_PATH, centroid)

        logger.info(
            "Successfully enrolled target using %d/%d images.",
            valid_count,
            len(image_paths),
        )
        logger.info("Target profile vector saved to: %s", TARGET_PROFILE_PATH)
    finally:
        detector.close()
        embedder.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="enroll",
        description="Enroll a single target subject from photos in data/raw_target/.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (detailed detector and enrollment logs).",
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    enroll_target()


if __name__ == "__main__":
    main()