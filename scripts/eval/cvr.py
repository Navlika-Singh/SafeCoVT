import json
import re
import ast
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
DATASET_ROOT = Path("/workspace/Qwen3-VL/cleaned")  # <-- change this
PREDICTIONS_JSON = Path("/workspace/Qwen3-VL/scripts/results/13_01_2025/CoT/Qwen2_5VL_3B_v3_likeCoVT_2026-02-01_00-44-32.json")  # <-- change this

# -----------------------------
# UTILS
# -----------------------------
def parse_bbox(bb_str):
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", bb_str)))
    if len(nums) < 4:
        raise ValueError(f"Bad bbox: {bb_str}")
    x1, y1, x2, y2 = nums[:4]
    return [max(0, x1), max(0, y1), min(1, x2), min(1, y2)]

def load_violating_boxes(index):
    meta_path = DATASET_ROOT / str(index) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    viol_raw = ast.literal_eval(meta["violating_objects_bounding_boxes"])
    viol_boxes = [parse_bbox(bb) for bb in viol_raw.values()]
    return viol_boxes

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def extract_points(output_text):
    pattern = re.compile(r"[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]")
    points = []
    for a, b in pattern.findall(output_text):
        x, y = float(a), float(b)
        if x > 1 or y > 1:
            x /= 1000
            y /= 1000
        points.append((x, y))
    return points

# -----------------------------
# METRIC
# -----------------------------
def compute_constraint_violation_rate(predictions_json_path):
    with open(predictions_json_path) as f:
        data = json.load(f)

    num_samples = len(data["samples"])
    num_violations = 0
    total_points = 0

    for sample in data["samples"]:

        if int(sample["index"]) < 5:
            continue

        index = int(sample["index"])
        output_text = sample["output"]
        points = extract_points(output_text)
        viol_boxes = load_violating_boxes(index)

        total_points += len(points)

        # if any point falls in a violating box -> count as violation
        if any(point_in_box(px, py, box) for px, py in points for box in viol_boxes):
            num_violations += 1

    cvr = num_violations / num_samples if num_samples > 0 else 0.0
    return cvr

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    cvr = compute_constraint_violation_rate(PREDICTIONS_JSON)
    print(f"Constraint Violation Rate (per sample): {cvr:.4f}")

