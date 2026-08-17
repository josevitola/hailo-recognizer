from hailo_platform import VDevice

_vdevice_instance: VDevice | None = None


def get_vdevice() -> VDevice:
    """
    Returns the process-wide shared Hailo VDevice instance.
    Lazy-instantiates the hardware handle on first call.
    """
    global _vdevice_instance
    if _vdevice_instance is None:
        _vdevice_instance = VDevice()
    return _vdevice_instance