from hailo_platform import HailoSchedulingAlgorithm, VDevice

_vdevice_instance: VDevice | None = None


def get_vdevice() -> VDevice:
    """
    Returns the process-wide shared Hailo VDevice instance.
    Lazy-instantiates the hardware handle on first call.

    Uses ROUND_ROBIN scheduling so multiple network groups (detector +
    embedder) can be activated simultaneously on the same device.
    """
    global _vdevice_instance
    if _vdevice_instance is None:
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        _vdevice_instance = VDevice(params)
    return _vdevice_instance