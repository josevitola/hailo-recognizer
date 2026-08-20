from pathlib import Path
import numpy as np
from hailo_platform import VDevice

from recognizer.base import HailoModelBase


class HailoEmbedder(HailoModelBase):
    def __init__(self, hef_path: str | Path, target: VDevice | None = None):
        super().__init__(hef_path, target)

    def extract_embedding(self, face_crop_rgb: np.ndarray) -> np.ndarray:
        """
        Resizes an RGB face crop to model input dimensions (112x112)
        and returns an L2-normalized 512-D embedding vector.
        """
        input_data = self._prepare_input(face_crop_rgb)

        # Execute on persistent stream pipeline
        results = self._infer(input_data)

        embedding = list(results.values())[0].flatten()

        # L2-normalize feature vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding