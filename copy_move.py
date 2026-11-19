"""Single-image repeated-region inspection using geometry and pixel verification.

Similar objects occur naturally. This produces copy-move evidence, never proof
of editing, and needs dataset validation before being used for app conclusions.
"""
import numpy as np
from PIL import Image


def inspect_copy_move(image, max_edge=1000):
    import cv2
    preview = image.convert("RGB")
    preview.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    rgb = np.asarray(preview).copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    points, descriptors = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.025).detectAndCompute(gray, None)
    empty = {"image": preview, "map": np.zeros(gray.shape, np.float32), "strength": 0., "regions": [],
             "analyzed_size": preview.size, "keypoints": len(points)}
    if descriptors is None or len(points) < 12:
        return empty
    neighbors = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, descriptors, k=min(8, len(points)))
    pairs = set()
    minimum_separation = max(16., min(gray.shape) * .04)
    for index, candidates in enumerate(neighbors):
        distant = [match for match in candidates if match.trainIdx != index and
                   np.linalg.norm(np.asarray(points[index].pt) - points[match.trainIdx].pt) >= minimum_separation]
        if len(distant) < 2 or distant[0].distance >= .7 * distant[1].distance:
            continue
        first, second = index, distant[0].trainIdx
        # Canonical spatial order prevents the reverse transform from being
        # counted as a second independent region pair.
        if points[first].pt > points[second].pt:
            first, second = second, first
        pairs.add((first, second))
    if len(pairs) < 6:
        return empty
    pairs = sorted(pairs)
    source = np.float32([points[i].pt for i, j in pairs])
    target = np.float32([points[j].pt for i, j in pairs])
    remaining = np.arange(len(pairs))
    found = []
    result_map = np.zeros(gray.shape, np.float32)
    for _ in range(6):
        if len(remaining) < 6:
            break
        cv2.setRNGSeed(20261119)
        transform, selected = cv2.estimateAffinePartial2D(source[remaining], target[remaining], method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.5, maxIters=3000, confidence=.995)
        if transform is None or selected is None:
            break
        keep = selected.ravel().astype(bool)
        chosen = remaining[keep]
        remaining = remaining[~keep]
        if len(chosen) < 6:
            continue
        a, b = source[chosen], target[chosen]
        scale = float(np.sqrt(np.linalg.det(transform[:, :2])))
        if not .5 <= scale <= 2 or np.median(np.linalg.norm(a - b, axis=1)) < minimum_separation:
            continue
        area_a = cv2.contourArea(cv2.convexHull(a))
        area_b = cv2.contourArea(cv2.convexHull(b))
        if min(area_a, area_b) < 60:
            continue
        warped = cv2.warpAffine(gray.astype(np.float32), transform, (gray.shape[1], gray.shape[0]))
        highpass = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.)
        warped_high = cv2.warpAffine(highpass, transform, (gray.shape[1], gray.shape[0]))
        correlations, residual_correlations = [], []
        for x, y in b:
            x, y = int(round(x)), int(round(y))
            if x < 8 or y < 8 or x + 8 >= gray.shape[1] or y + 8 >= gray.shape[0]:
                continue
            region = np.s_[y-8:y+9, x-8:x+9]
            def correlation(first, second):
                first = first.ravel() - first.mean(); second = second.ravel() - second.mean()
                denominator = np.linalg.norm(first) * np.linalg.norm(second)
                return float(np.dot(first, second) / denominator) if denominator > 1 else 0.
            correlations.append(correlation(warped[region], gray[region].astype(np.float32)))
            residual_correlations.append(correlation(warped_high[region], highpass[region]))
        if len(correlations) < 4:
            continue
        correlation = float(np.median(correlations))
        residual = float(np.median(residual_correlations))
        # This is an evidence index to be evaluated, not a probability.
        strength = max(0., min(correlation, residual)) * min(1., len(chosen) / 12)
        region = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(region, cv2.convexHull(a).astype(np.int32), 1)
        cv2.fillConvexPoly(region, cv2.convexHull(b).astype(np.int32), 1)
        result_map = np.maximum(result_map, region.astype(np.float32) * strength)
        found.append({"matches": len(chosen), "pixel_correlation": correlation, "residual_correlation": residual,
                      "strength": strength, "source_hull": cv2.convexHull(a).reshape(-1, 2).tolist(),
                      "target_hull": cv2.convexHull(b).reshape(-1, 2).tolist()})
    return {**empty, "map": result_map, "strength": max((region["strength"] for region in found), default=0.),
            "regions": found}
