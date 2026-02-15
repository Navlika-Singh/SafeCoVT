import json
import ast
import re
from pathlib import Path
import sys

# =========================
# CONFIG
# =========================

DATASET_ROOT = Path("/workspace/Qwen3-VL/cleaned")

JSON_A = "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_tooltoken_pre_icl_1000scale_xy_run4.json"

# List of other JSONs to compare against
JSON_OTHERS = [
    "/workspace/Qwen3-VL/scripts/results/13_01_2025/Baseline/Qwen3VL_4B_v1_2026-01-27_16-21-45.json",
    "/workspace/Qwen3-VL/scripts/results/13_01_2025/ICL/Qwen3VL_4B_v2_2026-01-27_19-02-28.json",
    "/workspace/Qwen3-VL/scripts/results/13_01_2025/CoTONLY/Qwen3VL_4B_v3_CoTONLY_2026-01-27_17-59-54.json.json",
    "/workspace/Qwen3-VL/scripts/results/13_01_2025/CoT/Qwen3VL_4B_v3_likeCoVT_2026-01-27_08-23-34.json",
    # add more JSON paths here
]

IOU_THRESHOLD = 0.05

SAVE_INDICES = True
OUTPUT_FILE = "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT_better_than_all_indices.json"


# =========================
# BBOX PARSING
# =========================

def parse_bbox(bb_str):
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", bb_str)))
    if len(nums) < 4:
        raise ValueError(f"Bad bbox: {bb_str}")
    x1, y1, x2, y2 = nums[:4]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(1, x2), min(1, y2)
    return [x1, y1, x2, y2]


# =========================
# IOU
# =========================

def compute_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter
    return 0 if union == 0 else inter / union


# =========================
# CLUSTER SAFE OBJECTS
# =========================

def filter_distinct_boxes(boxes, iou_thr=0.5):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for box in boxes:
        if all(compute_iou(box, k) < iou_thr for k in kept):
            kept.append(box)
    return kept


# =========================
# POINT EXTRACTION
# =========================

def extract_points(text):
    pat = re.compile(r"[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]")
    pts = []
    for a, b in pat.findall(text):
        x, y = float(a), float(b)
        if x > 1 or y > 1:
            x /= 1000
            y /= 1000
        pts.append((x, y))
    return pts


# =========================
# POINT IN BOX
# =========================

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


# =========================
# LOAD GT
# =========================

def load_gt(idx):
    meta_path = DATASET_ROOT / str(idx) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    safe_raw = ast.literal_eval(meta["non_violating_objects_bounding_boxes"])
    viol_raw = ast.literal_eval(meta["violating_objects_bounding_boxes"])
    safe_boxes = [parse_bbox(b) for b in safe_raw.values()]
    viol_boxes = [parse_bbox(b) for b in viol_raw.values()]
    safe_boxes = filter_distinct_boxes(safe_boxes, IOU_THRESHOLD)
    return safe_boxes, viol_boxes


# =========================
# METRIC
# =========================

def safe_region_coverage_score(sample, idx):
    points = extract_points(sample["output"])
    safe_boxes, viol_boxes = load_gt(idx)
    S = len(safe_boxes)

    # Violating hit => zero
    for px, py in points:
        for v in viol_boxes:
            if point_in_box(px, py, v):
                return 0.0

    # Edge cases
    if S == 0:
        return 1.0 if len(points) == 0 else 0.0
    if len(points) == 0:
        return 0.0

    # Coverage
    covered = 0
    for sb in safe_boxes:
        for px, py in points:
            if point_in_box(px, py, sb):
                covered += 1
                break
    return covered / S


# =========================
# COMPARATOR (GENERALIZED)
# =========================

def compare_multi(json_a, other_jsons):
    # Load A
    with open(json_a) as f:
        data_a = json.load(f)
    samples_a = data_a.get("samples", data_a)

    # Load all others
    all_others_samples = []
    for j in other_jsons:
        with open(j) as f:
            data = json.load(f)
        all_others_samples.append(data.get("samples", data))

    num_samples = len(samples_a)
    assert all(len(samples_a) == len(samples) for samples in all_others_samples), "Length mismatch!"

    better_indices = []

    for i in range(num_samples):
        try:
            score_a = safe_region_coverage_score(samples_a[i], i)
            scores_others = [safe_region_coverage_score(samples[i], i) for samples in all_others_samples]

            if all(score_a > s for s in scores_others):
                better_indices.append({
                    "index": i,
                    "score_A": round(score_a, 3),
                    "scores_others": [round(s, 3) for s in scores_others]
                })

        except Exception as e:
            print(f"[ERROR] Sample {i}: {e}")

    print("="*60)
    print("Model A better than all others")
    print("Total samples:", num_samples)
    print("A better count:", len(better_indices))
    print("="*60)

    return better_indices


# =========================
# RUN
# =========================

if __name__ == "__main__":
    better = compare_multi(JSON_A, JSON_OTHERS)

    if SAVE_INDICES:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(better, f, indent=2)
        print("Saved to:", OUTPUT_FILE)
