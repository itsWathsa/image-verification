# Comparing an image against an original

Upload the version to examine as the main image. In **Compare original**, select **I have an original/reference image**, upload the original, and click **Compare with original**.

The comparison is independent of Detectra and TruFor. It does not infer the reference's authenticity, identify the editing software, or establish whether a depicted event happened. It is useful when a known original exists; it cannot solve the same problem from a single unknown image.

## Method

Each decoded, orientation-corrected RGB image is reduced to at most 1000 pixels on its longest edge. SIFT feature matching in both directions and RANSAC estimate a homography. The comparison rejects insufficient matches, weak spatial coverage, implausible/mirrored transforms, excessive reprojection residual, or insufficient overlap. A homography assumes a shared geometric relationship; parallax and generative reconstruction can violate it even when some features align.

Only overlapping pixels are compared. A small fixed Gaussian blur reduces single-pixel noise, and a fixed per-channel absolute-difference setting marks pixels above 32 on the 0–255 scale. Small isolated components are removed. These are inspection settings, not trained or calibrated edit-detection thresholds. Non-overlapping borders appear gray and are excluded from the marked-area fraction. Marked area is **not an editing probability or a validated estimate of edited area**.

The interface displays the aligned reference, target, a plain difference map, and an optional red overlay. Plain maps avoid confusing an image's original colors with added highlights. Lighting, compression, resampling, geometric mismatch, and reconstruction may all produce differences.

## Bottle-removal example

The supplied Gemini version differs broadly from its camera original. The current alignment finds 139 inlier matches, about 0.84-pixel median residual, and about 94.65% overlap. Approximately 63% of the overlap is marked with the fixed inspection settings. This is **widespread reference differences**, not precise localization of only the removed bottle. Matching features cover a limited part of the image, so the app explicitly warns that alignment extrapolation may contribute to the result. No threshold was tuned to make this case pass.

The single-image TruFor detector scored the known edited version lower than its original. That failure remains documented; reference comparison does not retrospectively make the single-image detector correct.

## Checks

Tests cover identical images, resized/JPEG-recompressed copies, a controlled splice, unrelated or featureless inputs, and invalid transforms. These establish implementation behavior, not general forensic accuracy. Local input files and derived comparison maps remain excluded from Git.

Method reference: [OpenCV feature matching and homography](https://docs.opencv.org/4.13.0/d7/dff/tutorial_feature_homography.html).
