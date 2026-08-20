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


class HailoModelBase:
    """Base class for HailoRT models with persistent stream pipeline management."""

    def __init__(self, hef_path: str | Path, target: VDevice | None = None):
        self.hef = HEF(str(hef_path))

        # Use shared VDevice singleton if none passed explicitly
        self.target = target or get_vdevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_height = input_vstream_info.shape[0]
        self.input_width = input_vstream_info.shape[1]
        self.input_name = input_vstream_info.name

        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # Persistent stream pipeline (stream buffers allocated once)
        self.infer_pipeline = InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        )
        self.infer_pipeline.__enter__()

    def _infer(self, input_data: dict[str, np.ndarray]) -> dict:
        """
        Run inference on the persistent stream pipeline.

        The network group is activated/deactivated around each call because
        HailoRT only allows one activated network group per VDevice, and both
        detector and embedder share the same VDevice singleton.
        """
        with self.network_group.activate(self.network_group_params):
            return self.infer_pipeline.infer(input_data)

    def _prepare_input(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Resize image to model input dims and format as batch input."""
        resized = cv2.resize(image, (self.input_width, self.input_height))
        return {self.input_name: np.expand_dims(resized, axis=0)}

    def close(self) -> None:
        """Cleanly close persistent Hailo stream resources."""
        if hasattr(self, "infer_pipeline"):
            self.infer_pipeline.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()