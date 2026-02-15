import gradio as gr
import json
import ast
import re
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

# ===================== CONFIG =====================
BASE_DIR = "/workspace/Qwen3-VL/asimov_v2_constraints_with_rationale"  # <- set your base directory here
IMAGE_NAME = "image.jpg"
OUT_DIR_NAME = "cleaned"

# ===================== BB PARSING =====================
def parse_bb_string(bb_str):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", bb_str)
    if len(nums) != 4:
        raise ValueError(f"Malformed BB: {bb_str}")
    return tuple(map(float, nums))

def normalize_bb_field(field):
    if field in [None, "", {}]:
        return {}
    if isinstance(field, str):
        field = ast.literal_eval(field)
    if isinstance(field, dict):
        return field
    if isinstance(field, list):
        out = {}
        for item in field:
            if isinstance(item, dict) and "label" in item and "bbox" in item:
                out[item["label"]] = item["bbox"]
        return out
    return {}

# ===================== DRAWING =====================
def draw_bb(image, bbox, color, label):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1*w, y1*h, x2*w, y2*h], outline=color, width=4)
    draw.text((x1*w+4, y1*h+4), label, fill=color)
    return img

# ===================== DATA LOADING =====================
def load_sample(base_dir, idx):
    sample_dir = Path(base_dir) / str(idx)
    meta_path = sample_dir / "meta.json"
    image_path = sample_dir / IMAGE_NAME
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    image = Image.open(image_path).convert("RGB")
    violating = normalize_bb_field(meta.get("violating_objects_bounding_boxes"))
    non_violating = normalize_bb_field(meta.get("non_violating_objects_bounding_boxes"))
    bbs = []
    for label, bb_str in violating.items():
        try:
            bbs.append({"type":"violating","label":label,"bbox":parse_bb_string(bb_str),"raw":bb_str})
        except Exception as e:
            print(f"[SKIP] Violating BB '{label}': {e}")
    for label, bb_str in non_violating.items():
        try:
            bbs.append({"type":"non_violating","label":label,"bbox":parse_bb_string(bb_str),"raw":bb_str})
        except Exception as e:
            print(f"[SKIP] Non-violating BB '{label}': {e}")
    return image, meta, bbs

def save_cleaned(base_dir, idx, meta, kept):
    out_dir = Path(base_dir).parent / OUT_DIR_NAME / str(idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    v, nv = {}, {}
    for bb in kept:
        (v if bb["type"]=="violating" else nv)[bb["label"]] = bb["raw"]
    meta["violating_objects_bounding_boxes"] = str(v)
    meta["non_violating_objects_bounding_boxes"] = str(nv)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    shutil.copy(Path(base_dir)/str(idx)/IMAGE_NAME, out_dir/IMAGE_NAME)

# ===================== GRADIO LOGIC =====================
def init():
    idx = 0
    data = load_sample(BASE_DIR, idx)
    if data is None:
        raise gr.Error(f"No sample at idx 0 in {BASE_DIR}")
    image, meta, bbs = data
    img_bb, status_text = draw_bb(image, bbs[0]["bbox"], "red" if bbs[0]["type"]=="violating" else "green", bbs[0]["label"]), f"Sample {idx} | BB 1/{len(bbs)} ({bbs[0]['type']})"
    return img_bb, status_text, image, meta, bbs, idx, 0, [], BASE_DIR

def step(image, meta, bbs, idx, bb_idx, kept, base_dir, keep):
    if keep:
        kept.append(bbs[bb_idx])
    bb_idx += 1
    if bb_idx < len(bbs):
        bb = bbs[bb_idx]
        color = "red" if bb["type"]=="violating" else "green"
        return draw_bb(image, bb["bbox"], color, bb["label"]), f"Sample {idx} | BB {bb_idx+1}/{len(bbs)} ({bb['type']})", image, meta, bbs, idx, bb_idx, kept, base_dir
    save_cleaned(base_dir, idx, meta, kept)
    idx += 1
    data = load_sample(base_dir, idx)
    if data is None:
        return image, "All samples done", image, meta, bbs, idx, bb_idx, kept, base_dir
    image, meta, bbs = data
    bb = bbs[0]
    color = "red" if bb["type"]=="violating" else "green"
    return draw_bb(image, bb["bbox"], color, bb["label"]), f"Sample {idx} | BB 1/{len(bbs)} ({bb['type']})", image, meta, bbs, idx, 0, [], base_dir

# ===================== GRADIO UI =====================
with gr.Blocks() as demo:
    img = gr.Image()
    status = gr.Textbox()
    keep_btn = gr.Button("Keep")
    remove_btn = gr.Button("Remove")
    image_s = gr.State()
    meta_s = gr.State()
    bb_s = gr.State()
    idx_s = gr.State()
    bb_idx_s = gr.State()
    kept_s = gr.State()
    base_s = gr.State()
    
    # Initialize automatically
    img_out, status_out, image, meta, bbs, idx, bb_idx, kept, base_dir = init()
    img.value = img_out
    status.value = status_out
    image_s.value = image
    meta_s.value = meta
    bb_s.value = bbs
    idx_s.value = idx
    bb_idx_s.value = bb_idx
    kept_s.value = kept
    base_s.value = base_dir
    
    keep_btn.click(lambda *a: step(*a, keep=True),
                   inputs=[image_s, meta_s, bb_s, idx_s, bb_idx_s, kept_s, base_s],
                   outputs=[img, status, image_s, meta_s, bb_s, idx_s, bb_idx_s, kept_s, base_s])
    
    remove_btn.click(lambda *a: step(*a, keep=False),
                   inputs=[image_s, meta_s, bb_s, idx_s, bb_idx_s, kept_s, base_s],
                   outputs=[img, status, image_s, meta_s, bb_s, idx_s, bb_idx_s, kept_s, base_s])

demo.launch(share=True)

