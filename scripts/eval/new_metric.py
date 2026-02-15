import json
import ast
import re
from pathlib import Path
from collections import defaultdict


# =========================
# CONFIG
# =========================

DATASET_ROOT = Path("/workspace/Qwen3-VL/cleaned")

# MODEL_JSON = "/workspace/Qwen3-VL/scripts/results/13_01_2025/Baseline/Qwen3VL_4B_v1_2026-01-17_21-27-25.json"
# MODEL_JSON = "/workspace/Qwen3-VL/scripts/results/13_01_2025/ICL/Qwen3VL_4B_v2_2026-01-17_21-31-32.json"
# MODEL_JSON = "/workspace/Qwen3-VL/scripts/results/13_01_2025/CoTONLY/Qwen3VL_4B_v3_CoTONLY_2026-01-18_18-11-10.json"
# MODEL_JSON = "/workspace/Qwen3-VL/scripts/results/13_01_2025/CoT/Qwen3VL_4B_v3_2026-01-18_00-29-54.json"
MODEL_JSON = "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_tooltoken_pre_icl_1000scale_xy_run2.json"
# MODEL_JSON = "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_customprependPromptOnly.json"
# MODEL_JSON = "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_tooltoken_pre_icl_1000scale.json"

SAVE_LOW_SCORE_INDICES = False
OUTPUT_INDEX_FILE = "/workspace/Qwen3-VL/scripts/results/13_01_2025/ICL/meta/Qwen3VL_4B_v2_2026-01-27_19-02-28_FAILURE.json"
DEBUG = False

IOU_THRESHOLD = 0.05


# =========================
# BBOX PARSING
# =========================

def parse_bbox(bb_str):
    """
    Parses dataset bbox format:
    '[x: 0.33 y: 0.22, x: 0.55 y: 0.77]'
    Returns normalized [x1,y1,x2,y2]
    """

    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", bb_str)))

    if len(nums) < 4:
        raise ValueError(f"Bad bbox: {bb_str}")

    x1, y1, x2, y2 = nums[:4]

    # Clamp safety
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(1,x2), min(1,y2)

    return [x1, y1, x2, y2]


# =========================
# IOU
# =========================

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    inter = inter_w * inter_h

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union

def filter_distinct_boxes(boxes, iou_thr=0.5):
    """
    Keeps largest boxes when overlaps are high.
    Similar to Non-Max Suppression.
    """

    if not boxes:
        return []

    # Sort by area descending
    boxes = sorted(
        boxes,
        key=lambda b: (b[2]-b[0]) * (b[3]-b[1]),
        reverse=True
    )

    kept = []

    for box in boxes:
        keep = True

        for k in kept:
            if compute_iou(box, k) >= iou_thr:
                keep = False
                break

        if keep:
            kept.append(box)

    return kept

# =========================
# POINT EXTRACTION
# =========================

def extract_points(output_text):
    """
    Extracts points in (y,x) or (x,y) -> returns normalized [0-1]
    """

    pattern = re.compile(
        r"[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]"
    )

    points = []

    for a, b in pattern.findall(output_text):
        x, y = float(a), float(b)

        # normalize
        if x > 1 or y > 1:
            x /= 1000
            y /= 1000

        points.append((x, y))

    return points


# =========================
# POINT IN BOX
# =========================

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


# =========================
# LOAD DATASET GT
# =========================

def load_gt(index):
    meta_path = DATASET_ROOT / str(index) / "meta.json"

    with open(meta_path) as f:
        meta = json.load(f)

    safe_raw = ast.literal_eval(meta["non_violating_objects_bounding_boxes"])
    viol_raw = ast.literal_eval(meta["violating_objects_bounding_boxes"])

    safe_boxes = [parse_bbox(bb) for bb in safe_raw.values()]
    viol_boxes = [parse_bbox(bb) for bb in viol_raw.values()]

    safe_boxes = filter_distinct_boxes(safe_boxes, IOU_THRESHOLD)
    viol_boxes = filter_distinct_boxes(viol_boxes, IOU_THRESHOLD)

    return safe_boxes, viol_boxes


# =========================
# SCORE FUNCTION
# =========================

def safe_region_coverage_score(sample, index):
    points = extract_points(sample["output"])

    safe_boxes, viol_boxes = load_gt(index)

    S = len(safe_boxes)

    # ---- Rule 1: violating hit => zero
    for px, py in points:
        for vbox in viol_boxes:
            if point_in_box(px, py, vbox):
                if DEBUG:
                    print(f"[DEBUG] Index {index} -> Rule: Violating Hit | Score = 0.0")
                return 0.0, safe_boxes, viol_boxes

    # ---- Rule 2: empty GT handling
    if S == 0:
        # score = 1.0 if len(points) == 0 else 0.0
        score = 1.0
        if DEBUG:
            print(f"[DEBUG] Index {index} -> Rule: Empty Safe GT | Score = {score}")
        return score, safe_boxes, viol_boxes

    # ---- Rule 3: no predictions
    if len(points) == 0:
        if DEBUG:
            print(f"[DEBUG] Index {index} -> Rule: No Predicted Points | Score = 0.0")
        return 0.0, safe_boxes, viol_boxes

    # ---- Rule 4: coverage computation
    covered = 0

    for sbox in safe_boxes:
        hit = False

        for px, py in points:
            if point_in_box(px, py, sbox):
                hit = True
                break

        if hit:
            covered += 1

    score = covered / S

    if DEBUG:
        print(f"[DEBUG] Index {index} -> Rule: Coverage | Covered = {covered}/{S} | Score = {score:.3f}")

    return score, safe_boxes, viol_boxes


# =========================
# MAIN EVALUATION
# =========================

def evaluate_model(json_path):

    with open(json_path) as f:
        data = json.load(f)

    samples = data["samples"] if "samples" in data else data

    scores = []
    low_score_indices = []

    for idx, sample in enumerate(samples):
        if int(sample["index"]) < 5:
            continue
        s, safe_boxes, viol_boxes = safe_region_coverage_score(sample, idx)
        scores.append(s)

        # ---- Save indices with score < 1
        if s < 1.0:
            low_score_indices.append({
                "index": idx,
                "score": s,
                "safe_bbs": safe_boxes,
                "viol_boxes": viol_boxes
            })

    avg = sum(scores) / len(scores)

    print("=" * 50)
    print("Safe Region Coverage Score")
    print("=" * 50)
    print(f"Samples: {len(scores)}")
    print(f"Average: {avg:.4f}")
    print(f"Score < 1 count: {len(low_score_indices)}")
    print("=" * 50)

    # ---- Save to disk
    if SAVE_LOW_SCORE_INDICES:
        with open(OUTPUT_INDEX_FILE, "w") as f:
            json.dump(low_score_indices, f, indent=2)

        print(f"[Saved] Low-score indices written to:")
        print(OUTPUT_INDEX_FILE)

    return scores


# =========================
# RUN
# =========================

if __name__ == "__main__":

    evaluate_model(MODEL_JSON)
