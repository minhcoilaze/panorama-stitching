"""Domain-specific errors with messages suitable for the user interface."""


class PanoramaError(RuntimeError):
    """Base error for an expected panorama stitching failure."""


class InvalidImageError(PanoramaError):
    """Raised when an uploaded image cannot be decoded or validated."""


class InsufficientFeaturesError(PanoramaError):
    """Raised when an image contains too few usable local features."""


class InsufficientMatchesError(PanoramaError):
    """Raised when two images do not have a reliable overlap."""


class HomographyError(PanoramaError):
    """Raised when a stable projective transform cannot be estimated."""


class OrderingError(PanoramaError):
    """Raised when the automatic ordering graph is not reliable."""
