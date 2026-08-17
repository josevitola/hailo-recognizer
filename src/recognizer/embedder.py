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

        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

    def extract_embedding(self, face_crop_rgb: np.ndarray) -> np.ndarray:
        """
        Resizes an RGB face crop to model input dimensions (112x112)
        and returns an L2-normalized 512-D embedding vector.
        """
        resized_crop = cv2.resize(
            face_crop_rgb, (self.input_width, self.input_height)
        )
        input_data = {
            self.hef.get_input_vstream_infos()[0].name: np.expand_dims(
                resized_crop, axis=0
            )
        }

        with InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        ) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                results = infer_pipeline.infer(input_data)

        embedding = list(results.values())[0].flatten()

        # L2-normalize feature vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding