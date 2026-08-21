import logging
from pythonosc.udp_client import SimpleUDPClient
from recognizer.logging import get_logger

logger = get_logger(__name__) if "get_logger" in globals() else logging.getLogger(__name__)


class OSCSender:
    """Handles sending normalized target tracking coordinates over OSC."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 8000, address: str = "/eye/target"):
        self.ip = ip
        self.port = port
        self.address = address
        self.client = SimpleUDPClient(self.ip, self.port)
        logger.info("OSC Sender initialized targeting %s:%d (address: %s)", self.ip, self.port, self.address)

    def send_target_from_bbox(
        self, x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int
    ) -> tuple[float, float]:
        """
        Calculates bounding box center, normalizes to [-1.0, 1.0], and transmits OSC packet.
        """
        if frame_width <= 0 or frame_height <= 0:
            return 0.0, 0.0

        # Calculate bounding box center
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Normalize to [-1.0, 1.0] range
        # Note: X is flipped (1.0 - ...) so camera movement acts as a mirror relative to the screen
        norm_x = 1.0 - (2.0 * center_x / frame_width)
        norm_y = 1.0 - (2.0 * center_y / frame_height)

        # Transmit non-blocking UDP packet
        self.client.send_message(self.address, [float(norm_x), float(norm_y)])
        return norm_x, norm_y
