"""DH AG-95 configuration and lazily loaded driver."""

from .configuration_dh_ag95 import DHAG95GripperConfig

__all__ = ["DHAG95Gripper", "DHAG95GripperConfig"]


def __getattr__(name: str):
    if name == "DHAG95Gripper":
        from .dh_ag95 import DHAG95Gripper

        return DHAG95Gripper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
