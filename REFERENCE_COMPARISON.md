# Comparing an image against an original

Upload the version to examine as the main image. In **Compare original**, select **I have an original/reference image**, upload the original, and click **Compare with original**.

The comparison is independent of Detectra. It does not infer the reference's authenticity, identify the editing software, or establish whether a depicted event happened. It is useful when a known original exists; it cannot solve the same problem from a single unknown image.

## Method

Each decoded, orientation-corrected RGB image is reduced to at most 1000 pixels on its longest edge. SIFT feature matching in both directions and RANSAC estimate a homography. The comparison rejects insufficient matches, weak spatial coverage, implausible/mirrored transforms, excessive reprojection residual, or insufficient overlap. A homography assumes a shared geometric relationship; parallax and generative reconstruction can violate it even when some features align.

Only overlapping pixels are compared. A small fixed Gaussian blur reduces single-pixel noise, and a fixed per-channel absolute-difference setting marks pixels above 32 on the 0–255 scale. Small isolated components are removed. These are inspection settings, not trained or calibrated edit-detection thresholds. Non-overlapping borders appear gray and are excluded from the marked-area fraction. Marked area is **not an editing probability or a validated estimate of edited area**.

The interface displays the aligned reference, target, a plain difference map, and an optional red overlay. Plain maps avoid confusing an image's original colors with added highlights. Lighting, compression, resampling, geometric mismatch, and reconstruction may all produce differences.

## Brightness and contrast measurements

The comparison also measures display luma (0.299 R + 0.587 G + 0.114 B) over the valid, blurred overlap. It reports the candidate's mean minus the reference's mean on a 0-255 scale, and the candidate/reference ratio of luma standard deviations. Positive mean differences indicate a brighter examined image; ratios above one indicate a wider spread of light and dark values. If the reference's luma standard deviation is below two levels, the ratio is not reported because it is unstable.

An absolute mean shift of at least two levels or a contrast-spread change of at least 5% counts as a notable tonal difference for the comparison summary, even when no region exceeds the separate 32-level marking threshold. These are transparent inspection sensitivities, not learned forensic thresholds. Exposure, lighting, alignment and content changes can also affect these statistics; neither measurement identifies the software or proves an editing history. The combined JSON export retains these measurements.

## Colour measurements

For each valid, blurred RGB pixel, subtract its display luma from all three channels. Compare these brightness-removed channel values between the aligned original and candidate, compute the root mean square difference across channels, then average over valid pixels. Averaging per-pixel distances keeps opposite colour shifts in different regions from cancelling. A uniform additive brightness change without clipping yields zero colour difference; brightness scaling or clipping can also change colour balance. This measurement is a display-RGB inspection metric, not a perceptual Delta E value or a model probability.

An average colour difference of at least two levels contributes to the differences-detected summary independently of brightness, contrast and region marking. Mean signed red, green and blue shifts are also retained in the JSON report. Lighting, compression and imperfect alignment can affect the result. This sensitivity is an inspection setting, not a validated forensic threshold.

## Verification

Tests cover identical images, resized/JPEG-recompressed copies, a controlled splice, small brightness and colour adjustments below the region-marking threshold, contrast and saturation changes, opposite colour shifts, flat references, overlap exclusion, unrelated or featureless inputs, and invalid transforms. Streamlit tests check that a colour-only reference difference appears separately and survives JSON export. These establish implementation behavior, not general forensic accuracy. Local input files and derived comparison maps remain excluded from Git.

Method reference: [OpenCV feature matching and homography](https://docs.opencv.org/4.13.0/d7/dff/tutorial_feature_homography.html).
