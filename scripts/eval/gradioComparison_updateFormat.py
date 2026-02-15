import json
import ast
import random
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# =========================
# CONFIG
# =========================
DATASET_ROOT = Path("/workspace/Qwen3-VL/cleaned")

# MODEL_JSONS = {
#     "Qwen3VL-4B": "/workspace/Qwen3-VL/scripts/results/13_01_2025/Baseline/Qwen3VL_4B_v1_2026-01-17_21-27-25.json",
#     "Qwen3VL-4B-ICL": "/workspace/Qwen3-VL/scripts/results/13_01_2025/ICL/Qwen3VL_4B_v2_2026-01-17_21-31-32.json",
#     "Qwen3VL-4B-CoTOnly":"/workspace/Qwen3-VL/scripts/results/13_01_2025/CoTONLY/Qwen3VL_4B_v3_CoTONLY_2026-01-18_18-11-10.json",
#     "Qwen3VL-4B-CoT+ICL": "/workspace/Qwen3-VL/scripts/results/13_01_2025/CoT/Qwen3VL_4B_v3_likeCoVT_2026-01-27_08-23-34.json",
#     "Qwen3VL-4B-VCoT+ICL": "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_tooltoken_pre_icl_1000scale.json",
#     "Qwen3VL-4B-VCoT": "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_customprependPromptOnly.json"
#     }

MODEL_JSONS = {
    "Qwen3VL-4B-Baseline": "/workspace/Qwen3-VL/scripts/results/13_01_2025/Baseline/Qwen3VL_4B_v1_2026-01-27_16-21-45.json",
    "Qwen3VL-4B-ICL": "/workspace/Qwen3-VL/scripts/results/13_01_2025/ICL/Qwen3VL_4B_v2_2026-01-27_19-02-28.json",
    "Qwen3VL-4B-CoTOnly": "/workspace/Qwen3-VL/scripts/results/13_01_2025/CoTONLY/Qwen3VL_4B_v3_CoTONLY_2026-01-27_17-59-54.json.json",
    "Qwen3VL-4B-CoT": "/workspace/Qwen3-VL/scripts/results/13_01_2025/CoT/Qwen3VL_4B_v3_likeCoVT_2026-01-27_08-23-34.json",
    "Qwen3VL-4B-CoVT": "/workspace/Qwen3-VL/scripts/results/13_01_2025/VCoT/qwen3vl_4b_tooltoken_pre_icl_1000scale_xy_run4.json"
}

# MODEL_MARKERS = {
#     "Qwen3VL-4B": ("o", "blue"),
#     "Qwen3VL-4B-ICL": ("s", "purple"),
#     "Qwen3VL-4B-CoTOnly": ("D", "green"),
#     "Qwen3VL-4B-CoT+ICL": ("P", "orange"),
#     "Qwen3VL-4B-VCoT+ICL": ("^", "pink"),
#     "Qwen3VL-4B-VCoT": ("*", "yellow")
# }

MODEL_MARKERS = {
    "Qwen3VL-4B-Baseline": ("o", "blue"),
    "Qwen3VL-4B-ICL": ("s", "purple"),
    "Qwen3VL-4B-CoTOnly": ("D", "green"),
    "Qwen3VL-4B-CoT": ("P", "orange"),
    "Qwen3VL-4B-CoVT": ("*", "yellow")
}

# =========================
# LOAD MODEL DATA
# =========================

import re

def parse_bbox(bb_str):
    """
    Converts bbox string to normalized [x1,y1,x2,y2]
    """
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", bb_str)))

    if len(nums) < 4:
        raise ValueError(f"Invalid bbox: {bb_str}")

    x1, y1, x2, y2 = nums[:4]

    # Clamp to valid range
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(1, x2), min(1, y2)

    return [x1, y1, x2, y2]


def compute_iou(a, b):
    """
    IoU between two boxes
    """
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])

    union = areaA + areaB - inter

    return 0 if union == 0 else inter / union


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
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
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


def load_dataset_meta(index: int, iou_thr=0.5):
    """
    Loads GT bounding boxes from dataset meta.json
    and removes overlapping safe boxes.
    """

    meta_path = DATASET_ROOT / f"{index}" / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    sample = {}

    # ---------- SAFE BOXES ----------

    if "non_violating_objects_bounding_boxes" in meta:
        raw_safe = ast.literal_eval(
            meta["non_violating_objects_bounding_boxes"]
        )

        safe_boxes = [parse_bbox(b) for b in raw_safe.values()]
        safe_boxes = filter_distinct_boxes(safe_boxes, iou_thr)

        sample["non_violating_bbs"] = safe_boxes

    else:
        sample["non_violating_bbs"] = []

    # ---------- VIOLATING BOXES ----------

    if "violating_objects_bounding_boxes" in meta:
        raw_viol = ast.literal_eval(
            meta["violating_objects_bounding_boxes"]
        )

        viol_boxes = [parse_bbox(b) for b in raw_viol.values()]
        viol_boxes = filter_distinct_boxes(viol_boxes, iou_thr)

        # Normally DO NOT filter violating boxes (important to keep all)
        sample["violating_bbs"] = viol_boxes

    else:
        sample["violating_bbs"] = []

    # ---------- PROMPT ----------

    sample["prompt"] = meta.get("prompt", "[Prompt not found]")

    return sample


def safe_load_json(path):
    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                raise ValueError("Empty file")
            data = json.loads(content)

            # Pull the samples key if present
            if "samples" in data:
                return data["samples"]
            return data
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None



model_data = {}
for name, path in MODEL_JSONS.items():
    samples = safe_load_json(path)
    if samples is not None:
        # Convert bounding boxes strings → dicts
        # for sample in samples:
        #     if "non_violating_objects_bounding_boxes" in sample:
        #         sample["non_violating_bbs"] = ast.literal_eval(sample["non_violating_objects_bounding_boxes"])
        #     if "violating_objects_bounding_boxes" in sample:
        #         sample["violating_bbs"] = ast.literal_eval(sample["violating_objects_bounding_boxes"])
        model_data[name] = samples

NUM_SAMPLES = len(next(iter(model_data.values())))


# =========================
# IMAGE PATH MAPPING
# =========================
def index_to_image_path(index: int) -> Path:
    return DATASET_ROOT / f"{index}" / "image.jpg"


# =========================
# BOUNDING BOX UTILITIES
# =========================
import re

def parse_bb(bb_str, w, h):
    """
    Robustly parses ASIMOV bounding box strings.
    Extracts all floats in order: x1, y1, x2, y2
    """
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", bb_str)))

    if len(nums) < 4:
        raise ValueError(f"Could not parse bounding box: {bb_str}")

    x1, y1, x2, y2 = nums[:4]
    return (x1 * w, y1 * h, x2 * w, y2 * h)



def draw_bounding_boxes(image, sample):
    """
    Draws violating (red) and non-violating (green) bounding boxes
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for bb in sample["non_violating_bbs"]:

        x1, y1, x2, y2 = bb

        draw.rectangle(
            (x1 * w, y1 * h, x2 * w, y2 * h),
            outline="green",
            width=3
        )

    for bb in sample["violating_bbs"]:

        x1, y1, x2, y2 = bb

        draw.rectangle(
            (x1 * w, y1 * h, x2 * w, y2 * h),
            outline="red",
            width=3
        )

    return img


def draw_non_violating_only(image, sample):

    draw = ImageDraw.Draw(image)
    w, h = image.size

    for bb in sample["non_violating_bbs"]:

        x1, y1, x2, y2 = bb

        draw.rectangle(
            (x1 * w, y1 * h, x2 * w, y2 * h),
            outline="green",
            width=3
        )

    return image



def draw_violating_only(image, sample):
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for bb in sample["violating_bbs"]:

        x1, y1, x2, y2 = bb

        draw.rectangle(
            (x1 * w, y1 * h, x2 * w, y2 * h),
            outline="red",
            width=3
        )

    return image

# =========================
# OUTPUT PARSING
# =========================
def extract_points(output_text):
    points = []

    pattern = re.compile(
        r"[\[\(]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)]"
    )

    for x_str, y_str in pattern.findall(output_text):
        x, y = float(x_str), float(y_str)

        # If normalized, scale to 0–1000 (your plotting convention)
        if x <= 1.0 and y <= 1.0:
            x *= 1000
            y *= 1000

        points.append((x, y))

    return points

# =========================
# RENDER FUNCTION
# =========================
def render(index, selected_models):
    image_path = index_to_image_path(index)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # base_sample = next(iter(model_data.values()))[index]
    # prompt_text = base_sample.get("prompt", "[Prompt not found]")
    gt_sample = load_dataset_meta(index, iou_thr=0.5)
    prompt_text = gt_sample.get("prompt", "[Prompt not found]")

    # Three GT views
    # img_all = draw_bounding_boxes(image, base_sample)
    # img_non_violating = draw_non_violating_only(image, base_sample)
    # img_violating = draw_violating_only(image, base_sample)
    img_all = draw_bounding_boxes(image, gt_sample)
    img_non_violating = draw_non_violating_only(image, gt_sample)
    img_violating = draw_violating_only(image, gt_sample)

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.suptitle(
        f"PROMPT:\n{prompt_text}",
        fontsize=12,
        y=0.98,
        wrap=True,
    )

    axes[0].imshow(img_all)
    axes[0].set_title("GT: Violating + Non-Violating")
    axes[0].axis("off")

    axes[1].imshow(img_non_violating)
    axes[1].set_title("GT: Non-Violating Only (Green)")
    axes[1].axis("off")

    axes[2].imshow(img_violating)
    axes[2].set_title("GT: Violating Only (Red)")
    axes[2].axis("off")

    output_texts = {}

    for model in selected_models:
        sample = model_data[model][index]
        points = extract_points(sample["output"])
        marker, color = MODEL_MARKERS[model]

        for x, y in points:
            px = x * image.size[0] / 1000
            py = y * image.size[1] / 1000

            # Overlay predictions on ALL views
            for ax in axes:
                ax.scatter(
                    px,
                    py,
                    marker=marker,
                    s=120,
                    c=color,
                    label=model,
                )

        output_texts[model] = sample["output"]

    # Deduplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    return fig, output_texts


# =========================
# NAVIGATION
# =========================
def next_idx(i): return (i + 1) % NUM_SAMPLES
def prev_idx(i): return (i - 1) % NUM_SAMPLES

# =========================
# GRADIO UI
# =========================
with gr.Blocks() as demo:
    index_state = gr.State(0)

    gr.Markdown("## ASIMOV v2 – VLM Model Comparison")

    with gr.Row():
        back_btn = gr.Button("Back")
        forward_btn = gr.Button("Forward")

    # Index dropdown
    index_dropdown = gr.Dropdown(
        choices=list(range(NUM_SAMPLES)),
        value=0,
        label="Select Sample Index"
    )

    model_selector = gr.CheckboxGroup(
        choices=list(MODEL_JSONS.keys()),
        value=list(MODEL_JSONS.keys()),
        label="Select Models"
    )

    plot_out = gr.Plot(label="Image + Bounding Boxes + Model Outputs")
    output_out = gr.JSON(label="Model Output Text")

    def update(index, models):
        return render(index, models)

    # =========================
    # BUTTON NAVIGATION
    # =========================
    back_btn.click(
        lambda i: prev_idx(i),
        index_state,
        index_state
    ).then(
        lambda i: gr.update(value=i),
        index_state,
        index_dropdown
    ).then(
        update,
        [index_state, model_selector],
        [plot_out, output_out]
    )

    forward_btn.click(
        lambda i: next_idx(i),
        index_state,
        index_state
    ).then(
        lambda i: gr.update(value=i),
        index_state,
        index_dropdown
    ).then(
        update,
        [index_state, model_selector],
        [plot_out, output_out]
    )

    # =========================
    # DROPDOWN SELECTION
    # =========================
    index_dropdown.change(
        lambda i: i,
        index_dropdown,
        index_state
    ).then(
        update,
        [index_state, model_selector],
        [plot_out, output_out]
    )

    # =========================
    # MODEL TOGGLE
    # =========================
    model_selector.change(
        update,
        [index_state, model_selector],
        [plot_out, output_out]
    )

    # =========================
    # INITIAL LOAD
    # =========================
    demo.load(
        update,
        [index_state, model_selector],
        [plot_out, output_out]
    )

demo.launch(share=True)

