"""Core package for the panorama stitching application."""

from typing import Any

from .config import StitchConfig
from .exceptions import PanoramaError

__all__ = ["PanoramaError", "PanoramaStitcher", "StitchConfig", "StitchResult"]


def __getattr__(name: str) -> Any:
    """Delay importing OpenCV-heavy modules until the application needs them."""

    if name in {"PanoramaStitcher", "StitchResult"}:
        from .stitcher import PanoramaStitcher, StitchResult

        return {"PanoramaStitcher": PanoramaStitcher, "StitchResult": StitchResult}[name]
    raise AttributeError(name)
