"""Member 1 - Classical image processing:
hair removal, CLAHE, Otsu segmentation, ABCD features."""
import cv2
import numpy as np


def resize_max(img, max_side=512):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def remove_hair(img_bgr):
    """Blackhat morphology finds dark hairs, inpainting fills them in."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8))
    clean = cv2.inpaint(img_bgr, hair_mask, 1, cv2.INPAINT_TELEA)
    return clean, hair_mask


def enhance_clahe(img_bgr):
    """CLAHE on the L channel of LAB colour space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def segment_lesion(img_bgr):
    """Otsu threshold + morphology; keeps the main blob that does not touch the border."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    cnts = [c for c in cnts if cv2.contourArea(c) >= 0.01 * h * w]
    if not cnts:
        return np.zeros_like(mask)

    def touches_border(c):
        x, y, bw, bh = cv2.boundingRect(c)
        return x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1

    inner = [c for c in cnts if not touches_border(c)]
    best = max(inner or cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [best], -1, 255, thickness=-1)  # filled, no holes
    return clean


def abcd_features(img_bgr, mask):
    """Asymmetry, Border irregularity, Colour variation, Diameter."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50:
        return None
    perimeter = cv2.arcLength(c, True)
    h, w = mask.shape

    # A - Asymmetry: align principal axis, flip, measure non-overlap
    m = cv2.moments(mask, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    theta = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    M = cv2.getRotationMatrix2D((cx, cy), float(np.degrees(theta)), 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    aligned = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST) > 0
    a = max(int(aligned.sum()), 1)
    asym_h = np.logical_xor(aligned, aligned[:, ::-1]).sum() / a
    asym_v = np.logical_xor(aligned, aligned[::-1, :]).sum() / a
    asymmetry = float((asym_h + asym_v) / 2)

    # B - Border irregularity: compactness (1.0 = perfect circle) and solidity
    compactness = float(perimeter ** 2 / (4 * np.pi * area))
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = float(area / hull_area) if hull_area > 0 else 1.0

    # C - Colour variation: spread in LAB + number of distinct colour clusters
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    px = lab[mask > 0].astype(np.float32)
    color_std = float(px.std(axis=0).mean())
    if len(px) > 5000:
        px = px[np.random.RandomState(0).choice(len(px), 5000, replace=False)]
    k = 6
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(px)
    merged = []
    for i in range(k):
        if shares[i] >= 0.05 and all(np.linalg.norm(centers[i] - m_) > 25 for m_ in merged):
            merged.append(centers[i])
    n_colors = len(merged)

    # D - Diameter (pixels; real mm needs the camera scale, so also give it relative to image)
    diameter_px = float(np.sqrt(4 * area / np.pi))
    diameter_rel = diameter_px / max(h, w)

    flags = {
        "Asymmetry": asymmetry > 0.20,
        "Border irregular": compactness > 1.30,
        "Colour variation": n_colors >= 3,
        "Large diameter": diameter_rel > 0.40,
    }
    return {
        "asymmetry": round(asymmetry, 3),
        "compactness": round(compactness, 3),
        "solidity": round(solidity, 3),
        "color_std": round(color_std, 2),
        "n_colors": n_colors,
        "diameter_px": round(diameter_px, 1),
        "diameter_rel": round(diameter_rel, 3),
        "flags": flags,
        "abcd_score": int(sum(flags.values())),
    }


def analyze(img_bgr):
    """Full classical pipeline. Returns all intermediate images + ABCD features."""
    img = resize_max(img_bgr)
    clean, hair_mask = remove_hair(img)
    enhanced = enhance_clahe(clean)
    mask = segment_lesion(enhanced)
    overlay = enhanced.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)
    feats = abcd_features(clean, mask)
    return {"original": img, "hair_mask": hair_mask, "clean": clean,
            "enhanced": enhanced, "mask": mask, "overlay": overlay, "features": feats}
