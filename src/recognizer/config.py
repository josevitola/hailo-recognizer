import os
from pathlib import Path

# Resolve absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load paths from environment variables, falling back to default .hef paths
DETECTOR_MODEL_PATH = Path(
    os.getenv("DETECTOR_MODEL_PATH", PROJECT_ROOT / "models" / "scrfd_2.5g_h8.hef")
)

EMBEDDER_MODEL_PATH = Path(
    os.getenv("EMBEDDER_MODEL_PATH", PROJECT_ROOT / "models" / "arcface_mobilefacenet.hef")
)

TARGET_PROFILE_PATH = Path(
    os.getenv("TARGET_PROFILE_PATH", PROJECT_ROOT / "data" / "target_profile.npy")
)

RAW_TARGET_DIR = Path(
    os.getenv("RAW_TARGET_DIR", PROJECT_ROOT / "data" / "raw_target")
)

ACCEPTED_DIR = Path(
    os.getenv("ACCEPTED_DIR", PROJECT_ROOT / "data" / "accepted")
)

DETECTOR_THRESHOLD = float(os.getenv("DETECTOR_THRESHOLD", 0.5))
MATCHER_THRESHOLD = float(os.getenv("MATCHER_THRESHOLD", 0.55))