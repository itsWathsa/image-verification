"""Reference-based image differences, not a single-image authenticity detector."""

import numpy as np
from PIL import Image


def _checked_transform(transform, source_shape, target_shape):
    """Reject folded, singular, or extreme projections before warping pixels."""
    import cv2

    if transform is None or transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("Inconclusive: image alignment failed.")
    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    source_scale = np.diag([source_w, source_h, 1.])
    target_scale = np.diag([1. / target_w, 1. / target_h, 1.])
    normalized = target_scale @ transform @ source_scale
    if np.linalg.cond(normalized) > 10000:
        raise ValueError("Inconclusive: unstable image alignment.")
    corners = np.float32([[0, 0], [source_w - 1, 0],
                          [source_w - 1, source_h - 1], [0, source_h - 1]])
    denominator = np.c_[corners, np.ones(4)] @ transform[2]
    if np.min(np.abs(denominator)) < 1e-8 or not (np.all(denominator > 0) or np.all(denominator < 0)):
        raise ValueError("Inconclusive: image alignment crosses an invalid perspective boundary.")
    projected = cv2.perspectiveTransform(corners[None], transform)[0]
    contour = projected.reshape(-1, 1, 2)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(contour):
        raise ValueError("Inconclusive: implausible image alignment.")
    area = cv2.contourArea(contour, oriented=True)
    area_ratio = area / (target_w * target_h)
    if not .15 <= area_ratio <= 8:
        raise ValueError("Inconclusive: alignment is mirrored or has an extreme scale change.")
    return float(area_ratio)


def compare_reference(original, candidate):
    import cv2

    def pixels(image):
        preview = image.convert("RGB")
        preview.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        return np.asarray(preview).copy()

    reference, target = pixels(original), pixels(candidate)
    height, width = target.shape[:2]
    if reference.shape == target.shape and np.array_equal(reference, target):
        aligned = reference
        valid = np.ones((height, width), dtype=bool)
        diagnostics = {"method": "identical decoded pixels", "inliers": None, "inlier_ratio": None}
    else:
        sift = cv2.SIFT_create(nfeatures=4000)
        key_a, desc_a = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
        key_b, desc_b = sift.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), None)
        if desc_a is None or desc_b is None:
            raise ValueError("Inconclusive: too little texture to align these images reliably.")
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(desc_a, desc_b, k=2)
        reverse = matcher.knnMatch(desc_b, desc_a, k=2)
        reverse_pairs = {(pair[0].trainIdx, pair[0].queryIdx) for pair in reverse
                         if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance}
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance
                and (pair[0].queryIdx, pair[0].trainIdx) in reverse_pairs]
        if len(good) < 12:
            raise ValueError("Inconclusive: not enough reliable image correspondences.")
        a = np.float32([key_a[item.queryIdx].pt for item in good])
        b = np.float32([key_b[item.trainIdx].pt for item in good])
        transform, inliers = cv2.findHomography(a, b, cv2.RANSAC, 3.)
        if inliers is None:
            raise ValueError("Inconclusive: image alignment failed.")
        area_ratio = _checked_transform(transform, reference.shape, target.shape)
        keep = inliers.ravel().astype(bool)
        if keep.sum() < 12 or keep.mean() < .35:
            raise ValueError("Inconclusive: image alignment is too uncertain.")
        source_coverage = cv2.contourArea(cv2.convexHull(a[keep])) / (reference.shape[0] * reference.shape[1])
        target_coverage = cv2.contourArea(cv2.convexHull(b[keep])) / (width * height)
        if min(source_coverage, target_coverage) < .08:
            raise ValueError("Inconclusive: matching features cover too little of the image.")
        projected = cv2.perspectiveTransform(a[keep][None], transform)[0]
        errors = np.linalg.norm(projected - b[keep], axis=1)
        median_error = float(np.median(errors))
        if median_error > 2.5:
            raise ValueError("Inconclusive: alignment residuals are too large.")
        aligned = cv2.warpPerspective(reference, transform, (width, height))
        support = cv2.warpPerspective(np.ones(reference.shape[:2], np.uint8), transform, (width, height), flags=cv2.INTER_NEAREST)
        # Explicit zero border excludes unsupported edges from blur/differences.
        valid = cv2.erode(support, np.ones((7, 7), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if valid.mean() < .6:
            raise ValueError("Inconclusive: insufficient overlapping image area after alignment.")
        diagnostics = {"method": "mutual SIFT matches + RANSAC homography", "inliers": int(keep.sum()),
                       "inlier_ratio": float(keep.mean()), "mutual_matches": len(good),
                       "reference_feature_coverage": float(source_coverage),
                       "candidate_feature_coverage": float(target_coverage),
                       "spatial_support": "limited" if min(source_coverage, target_coverage) < .2 else "broad",
                       "median_reprojection_error_pixels": median_error,
                       "projected_reference_area_ratio": area_ratio}
    # Fixed display/inspection settings, not trained edit-detection thresholds.
    first = cv2.GaussianBlur(aligned, (5, 5), 1.)
    second = cv2.GaussianBlur(target, (5, 5), 1.)
    difference = np.max(np.abs(first.astype(np.float32) - second.astype(np.float32)), axis=2)
    difference[~valid] = 0
    mask = ((difference > 32) & valid).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 32:
            cleaned[labels == index] = 1
    overlay = target.copy()
    marked = cleaned.astype(bool)
    overlay[marked] = (.4 * overlay[marked] + .6 * np.array([255, 0, 0])).astype(np.uint8)
    overlay[~valid] = 128
    aligned[~valid] = 128
    difference_display = np.round(difference).astype(np.uint8)
    difference_display[~valid] = 128
    changed_fraction = float(cleaned.sum() / valid.sum())
    status = ("widespread reference differences" if changed_fraction > .35 else
              "localized reference differences" if changed_fraction else "no marked reference differences")
    return {"aligned_reference": Image.fromarray(aligned), "candidate": Image.fromarray(target),
            "overlay": Image.fromarray(overlay), "difference": Image.fromarray(difference_display),
            "mask": Image.fromarray(cleaned * 255), "changed_fraction": float(cleaned.sum() / valid.sum()),
            "overlap_fraction": float(valid.mean()), "analysis_size": (width, height),
            "overlap_mask": Image.fromarray(valid.astype(np.uint8) * 255),
            "comparison_status": status,
            "display_notes": "Gray borders were excluded. Marking settings are exploratory inspection aids, not validated edit-detection thresholds.",
            "inspection_settings": {"maximum_edge": 1000, "difference_threshold_0_to_255": 32,
                                    "minimum_component_pixels": 32, "gaussian_sigma": 1.},
            "alignment": diagnostics, "assessment": "reference differences only; authenticity inconclusive"}
