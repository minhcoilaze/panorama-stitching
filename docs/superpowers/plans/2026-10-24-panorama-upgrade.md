# Panorama Upgrade Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans;
> use superpowers:test-driven-development and one final independent review.

**Goal:** Three projection modes, custom core algorithms, closed crop and 100/150 MP export.
**Architecture:** Pure NumPy geometry/optimization, draft masks/seams, original-image tiled renderer.
**Tech Stack:** Python 3.11, NumPy, OpenCV only IO/SIFT/ORB, Streamlit.
**Spec:** ../specs/2026-10-24-panorama-upgrade-design.md

## Global Constraints
No product calls to Stitcher/findHomography/warpPerspective/remap/BFMatcher,
library optimizers, seam finders or blenders. Never silently downscale target MP.

## Review Focus
- Degenerate or non-overlapping images: clear error, no nonsensical large render.
- Cylindrical angle wrap/behind camera: finite boundaries and valid inverse map.
- Internal mask holes and bilinear border: cropped export must be entirely valid.
- Native versus requested megapixels: scale transformations once, preserve originals.
- Large output lifetime/UI reruns: temp cleanup, persistent results, bounded previews.

## Tasks (each test is written and observed failing before implementation)
- [x] 1. Custom matching and robust optimizer; test known nearest neighbors/outliers.
- [x] 2. Camera focal/rotation recovery and projections; synthetic recovery/round trips.
- [x] 3. Automatic selection, largest rectangle and graph-cut; known geometry/mask/seam tests.
- [x] 4. Tiled rendering and output policies; dense-vs-tiled, all-valid crop, target MP, cleanup.
- [x] 5. Pipeline and Streamlit integration; original tests, shuffled/invalid modes, app test.
- [x] 6. Real image runs, 100/150 MP encode, visual QA, documentation, final review.

## Verification
Use .venv/Scripts/python.exe -B -m pytest -p no:cacheprovider.
Use scripts/diagnose_dataset.py for real inputs and output measurements.
No git commits/worktree because this directory is not a Git repository.
