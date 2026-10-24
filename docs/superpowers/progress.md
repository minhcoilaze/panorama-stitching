# Execution ledger — plan: plans/2026-10-24-panorama-upgrade.md

- User approved design and requested implementation with custom core algorithms.
- Ruling: work in place with ZIP source backup; no Git repository exists.
- Ruling: SIFT/ORB remain library primitives per approved pipeline; matching,
  geometry, optimizer, sampling, seams, blending and crop are implemented here.
- Pre-flight: camera model/projector shared by draft analysis and final renderer;
  output coordinates always in registration space, original scales applied once.
- Pre-flight: result metadata/UI use the same projection/reason/resolution fields.

- Baseline: 7 passed (19.62 s).
- Tasks 1–4: RED imports/behaviors observed before implementation; 25 tests passed after pipeline integration (18.55 s).
- Task 5 UI: missing controls RED; new controls/AppTest GREEN (1 passed, 1.22 s).
- Custom camera synthetic: known focal 800 px recovered within 3%, round trips passed.
- Real images: custom SIFT matches reproduce the original pair counts; refined cylindrical fit approximately 3 px.
- Native real preview rendered and visually inspected; no blank borders in the closed crop.
- Ruling: use shared normalized focal for fixed-lens sweeps; reject unidentifiable translation instead of fabricating calibration.
- Ruling: use threadpoolctl solely to bound BLAS threads (4) for predictable matching/optimizer performance; no packaged core algorithms added.
- Large-export benchmark in progress.

- Independent review: no Critical; 2 Important findings confirmed with RED tests.
- Final: fixed resize rounding drift — test_original_sampling_uses_actual_per_axis_resize_scales RED→GREEN.
- Final: fixed inconsistent global alignment acceptance — two bad-planar/cycle tests RED→GREEN; per-edge symmetric validation now checks both camera and planar candidates.
- Ruling: use Numba only as a compiler for the hand-written fused scalar renderer; retain NumPy reference and compare within 1 intensity level for both projections. Cost if unavailable: NumPy fallback is correct but slower.
- Large real export before acceleration: 100.019709 MP, 19,857×5,037, closed coverage verified, 348.72 s, peak 801.39 MB.
- UI real-upload flow tested: actual synthetic stitching, switching PNG→JPG retains the same result and signature (2 AppTests passed).
- Min-cut independently checked on 48 random tiny directed graphs against exhaustive enumeration.

- Task 6: completed actual Auto 100 MP JPEG and forced Cylindrical 150 MP PNG, both full-resolution coverage checks passed. PNG/JPEG headers independently verified.
- Task 6: completed forced Planar native real-image run; distortion warning and closed crop behave as intended.
- Final suite: 34 passed in 21.62 s, including real Streamlit upload/rerun flow and compiled-vs-NumPy reference tests.
- Environment: pip check → No broken requirements found. No panorama-*.dat files left after completed diagnostic processes.
- Final review: independent read-only review performed; both Important findings fixed with RED→GREEN tests and green full suite. No Critical findings; no deferred Minor findings.
- Remaining model limits: fixed-lens rotating-camera model, parallax/motion may still produce residual seams; 100/150 MP may require interpolation. These are exposed in UI and README.
