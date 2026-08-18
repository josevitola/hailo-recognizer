# Task Specification: Persistent HailoRT Pipeline Optimization

Implement persistent pipeline activation for both `HailoFaceDetector` and `HailoEmbedder` to fix extreme inference lag during live video streams.

---

## Objective

Currently, `detector.py` and `embedder.py` create and destroy `InferVStreams` context managers and call `network_group.activate()` on every single inference call. Re-allocating HailoRT stream buffers per frame incurs massive PCIe bus overhead, reducing frame rates from 30+ FPS down to 0.1 FPS.

Refactor both classes to activate the Hailo `network_group` and instantiate `InferVStreams` **once during `__init__`**, reusing the open pipeline for all subsequent inference calls.

---

## 1. Updates to `src/recognizer/embedder.py`

### Changes Required
1. Activate `network_group` in `__init__` and store the context handle.
2. Initialize `InferVStreams` once in `__init__` and keep the pipeline open.
3. Update `extract_embedding()` to use `self.infer_pipeline.infer(...)` directly without local `with` blocks.
4. Add a `close()` method (and `__enter__`/`__exit__` support) to cleanly release HailoRT resources on shutdown.

### Proposed Implementation

```python
from pathlib import Path
import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

from recognizer.device import get_vdevice


class HailoEmbedder:
    def __init__(self, hef_path: str | Path, target: VDevice | None = None):
        self.hef = HEF(str(hef_path))
        self.target = target or get_vdevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_height = input_vstream_info.shape[0]
        self.input_width = input_vstream_info.shape[1]

        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # Persistent activation and stream pipeline
        self._activated_network_group = self.network_group.activate(
            self.network_group_params
        )
        self._activated_network_group.__enter__()

        self.infer_pipeline = InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        )
        self.infer_pipeline.__enter__()

    def extract_embedding(self, face_crop_rgb: np.ndarray) -> np.ndarray:
        resized_crop = cv2.resize(
            face_crop_rgb, (self.input_width, self.input_height)
        )
        input_data = {
            self.hef.get_input_vstream_infos()[0].name: np.expand_dims(
                resized_crop, axis=0
            )
        }

        # Direct execution on persistent stream
        results = self.infer_pipeline.infer(input_data)
        embedding = list(results.values())[0].flatten()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def close(self) -> None:
        """Cleanly close persistent Hailo stream resources."""
        if hasattr(self, "infer_pipeline"):
            self.infer_pipeline.__exit__(None, None, None)
        if hasattr(self, "_activated_network_group"):
            self._activated_network_group.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

---

## 2. Updates to `src/recognizer/detector.py`

### Changes Required
1. Mirror the persistent context pattern in `HailoFaceDetector.__init__`.
2. Update `detect()` to execute `self.infer_pipeline.infer(input_data)` directly without per-frame stream instantiation.
3. Add a `close()` method to cleanup `InferVStreams` and `network_group` contexts.

### Proposed Implementation

```python
class HailoFaceDetector:
    def __init__(
        self,
        hef_path: str | Path,
        confidence_threshold: float = 0.60,
        nms_threshold: float = 0.40,
        target: VDevice | None = None,
    ):
        self.hef = HEF(str(hef_path))
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target = target or get_vdevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_height = input_vstream_info.shape[0]
        self.input_width = input_vstream_info.shape[1]

        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        self.strides = [8, 16, 32]
        self.anchors_by_stride = self._generate_anchors()

        # Persistent activation and stream pipeline
        self._activated_network_group = self.network_group.activate(
            self.network_group_params
        )
        self._activated_network_group.__enter__()

        self.infer_pipeline = InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        )
        self.infer_pipeline.__enter__()

    def detect(self, image_rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        orig_h, orig_w, _ = image_rgb.shape
        resized_frame = cv2.resize(
            image_rgb, (self.input_width, self.input_height)
        )
        input_data = {
            self.hef.get_input_vstream_infos()[0].name: np.expand_dims(
                resized_frame, axis=0
            )
        }

        # Direct execution on persistent stream
        raw_results = self.infer_pipeline.infer(input_data)

        # ... (rest of parsing, decoding, and NMS logic remains unchanged) ...

    def close(self) -> None:
        """Cleanly close persistent Hailo stream resources."""
        if hasattr(self, "infer_pipeline"):
            self.infer_pipeline.__exit__(None, None, None)
        if hasattr(self, "_activated_network_group"):
            self._activated_network_group.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

---

## 3. Updates to `src/recognizer/__init__.py`

Ensure proper cleanup of Hailo resources on application exit:

```python
try:
    while True:
        # Main camera capture & inference loop
        ...
finally:
    detector.close()
    embedder.close()
    picam2.stop()
    cv2.destroyAllWindows()
```

---

## Verification Criteria
- [ ] Run `uv run preview` and verify video stream frame rate increases to smoothly render live video.
- [ ] Ensure quitting via `q` terminates cleanly without Hailo driver hanging errors.
