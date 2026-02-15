import json
from pathlib import Path

import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

# =========================
# CONFIG
# =========================
DATASET_NAME = "/rds/general/user/ns1324/home/iso/data/content/data/asimov_v2_constraints_with_rationale/0.1.0"
SPLIT = "val"
BASE_DIR = Path("/rds/general/user/ns1324/home/iso/data/asimov_v2_constraints_with_rationale")

BASE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATASET
# =========================
builder = tfds.core.read_only_builder.ReadOnlyBuilder(
    builder_dir=DATASET_NAME
)
ds = builder.as_dataset(split=SPLIT)

# =========================
# ITERATE + SAVE
# =========================
for idx, example in enumerate(tfds.as_numpy(ds)):
    example_dir = BASE_DIR / str(idx)
    example_dir.mkdir(parents=True, exist_ok=True)

    meta = {}

    # ---------- IMAGE (single) ----------
    if "image" in example:
        image_array = example["image"]
        if isinstance(image_array, np.ndarray) and image_array.ndim == 3:
            Image.fromarray(image_array).save(
                example_dir / "image.jpg",
                format="JPEG"
            )

    # ---------- VIDEO ----------
    if "video" in example:
        video = example["video"]

        frames = video.get("frames", None)
        timestamps = video.get("timestamps", None)

        if frames is not None:
            frames_dir = example_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(
                    frames_dir / f"{i:06d}.jpg",
                    format="JPEG"
                )

        # Store timestamps in metadata (decoded)
        if timestamps is not None:
            meta["video_timestamps"] = [
                ts.decode("utf-8") if isinstance(ts, (bytes, bytearray)) else ts
                for ts in timestamps
            ]

    # ---------- METADATA ----------
    for key, value in example.items():
        if key in {"image", "video"}:
            continue

        if isinstance(value, (bytes, bytearray)):
            meta[key] = value.decode("utf-8")
        else:
            meta[key] = value

    with open(example_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if idx % 5 == 0:
        print(f"Processed {idx} examples")

print("Done.")
