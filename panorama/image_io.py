"""Safe image decoding and result encoding for the user interface."""

from __future__ import annotations

import cv2
import numpy as np

from .exceptions import InvalidImageError


def decode_image(data: bytes, filename: str = "image") -> np.ndarray:
    if not data:
        raise InvalidImageError(f"{filename}: the uploaded file is empty.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    # IMREAD_COLOR normalizes grayscale/alpha inputs to BGR and, unlike
    # IMREAD_UNCHANGED, lets OpenCV apply EXIF orientation on phone photos.
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(f"{filename}: OpenCV could not decode this file.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"{filename}: unsupported channel count.")
    if image.shape[0] * image.shape[1] > 60_000_000:
        raise InvalidImageError(
            f"{filename}: image exceeds the 60-megapixel safety limit."
        )
    return image


def encode_image(image: np.ndarray, extension: str = ".png", quality: int = 95) -> bytes:
    extension = extension.lower()
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Output extension must be PNG or JPEG.")
    parameters: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        if max(image.shape[:2]) > 65500:
            raise InvalidImageError('JPEG supports sides up to 65500 pixels. Export this panorama as PNG.')
        parameters = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))]
    try:
        success, encoded = cv2.imencode(extension, image, parameters)
    except cv2.error as error:
        raise InvalidImageError(f'The panorama could not be encoded: {error}') from error
    if not success:
        raise InvalidImageError("The panorama could not be encoded for download.")
    return encoded.tobytes()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
