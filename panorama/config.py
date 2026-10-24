"""Configuration values shared by the stitching pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class StitchConfig:
    """Runtime configuration for feature matching and image composition."""

    feature_method: str = "SIFT"
    ratio_threshold: float = 0.75
    min_matches: int = 10
    ransac_threshold: float = 4.0
    ransac_confidence: float = 0.995
    ransac_max_iterations: int = 3000
    max_dimension: int = 1920
    feather_power: float = 1.0
    exposure_compensation: bool = True
    random_seed: int = 42
    projection: str = 'auto'
    output_megapixels: float | None = 100.0
    auto_crop: bool = True
    draft_dimension: int = 1000
    seam_dimension: int = 160
    tile_rows: int = 96
    tile_columns: int = 1024

    def validate(self) -> None:
        if self.projection not in {'auto', 'planar', 'cylindrical'}:
            raise ValueError('projection must be auto, planar or cylindrical.')
        if self.output_megapixels is not None and (not isfinite(self.output_megapixels) or not 0 < self.output_megapixels <= 150):
            raise ValueError('output_megapixels must be positive and at most 150, or None for native resolution.')
        if self.draft_dimension < 100 or self.seam_dimension < 20:
            raise ValueError('Draft/seam dimensions are too small.')
        if min(self.tile_rows, self.tile_columns) < 1:
            raise ValueError('Tile dimensions must be positive.')
        method = self.feature_method.upper()
        if method not in {"SIFT", "ORB"}:
            raise ValueError("feature_method must be either 'SIFT' or 'ORB'.")
        if not 0.5 <= self.ratio_threshold <= 0.95:
            raise ValueError("ratio_threshold must be between 0.5 and 0.95.")
        if self.min_matches < 4:
            raise ValueError("min_matches must be at least 4.")
        if not isfinite(self.ransac_threshold) or self.ransac_threshold <= 0:
            raise ValueError("ransac_threshold must be positive.")
        if not 0 < self.ransac_confidence < 1:
            raise ValueError("ransac_confidence must be between 0 and 1.")
        if self.ransac_max_iterations < 1:
            raise ValueError("ransac_max_iterations must be positive.")
        if self.max_dimension < 320:
            raise ValueError("max_dimension must be at least 320 pixels.")
        if not isfinite(self.feather_power) or self.feather_power <= 0:
            raise ValueError("feather_power must be positive.")
