import numpy as np
from pathlib import Path

class TargetMatcher:
    def __init__(self, profile_path: str | Path, threshold: float = 0.45):
        self.threshold = threshold
        self.profile_path = Path(profile_path)

        if not self.profile_path.exists():
            raise FileNotFoundError(
                f"Target profile not found at {self.profile_path}. "
                "Run 'uv run python -m recognizer.enroll' first."
            )

        # Load pre-computed 512-D target centroid into RAM
        self.target_centroid = np.load(str(self.profile_path))

    def match(self, live_embedding: np.ndarray) -> tuple[bool, float]:
        """
        Calculates cosine similarity between a live face vector and the target profile.
        Returns: (is_target_match, similarity_score)
        """
        # Vector dot product equivalent to cosine similarity for normalized vectors
        similarity = float(np.dot(live_embedding, self.target_centroid))
        is_match = similarity >= self.threshold

        return is_match, similarity