import os
import json
import gc
import time
import argparse
from pathlib import Path
from datetime import datetime
import re

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# Disable TF GPU
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

# =========================
# HARD-CODED PATHS ONLY
# =========================
DATASET_DIR = "/rds/general/user/ns1324/home/iso/data/asimov_v2_constraints_with_rationale"
PROMPTS_FILE = "/rds/general/user/ns1324/home/iso/scripts/prompts.json"

# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)

    parser.add_argument("--prompt_version", required=True)
    parser.add_argument("--prompt_position", choices=["prepend", "append", "none"], default="none")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--output_dir", required=True)

    return parser.parse_args()


# =========================
# DATASET LOADER
# =========================
def load_dataset(base_dir: str):
    data = []
    base = Path(base_dir)

    for sample_dir in sorted(base.iterdir(), key=lambda x: int(x.name)):
        image_path = sample_dir / "image.jpg"
        meta_path = sample_dir / "meta.json"

        if not image_path.exists() or not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        image = np.array(Image.open(image_path).convert("RGB"))

        data.append({
            "id": sample_dir.name,
            "image": image,
            "meta": meta
        })

    return data


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(PROMPTS_FILE) as f:
        prompt_dict = json.load(f)

    base_prompt = prompt_dict[args.prompt_version]

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_path = Path(args.output_dir) / f"{args.model_name}_{args.prompt_version}_{run_time}.json"

    # =========================
    # SEEDING
    # =========================
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tf.random.set_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =========================
    # MODEL
    # =========================
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    dataset = load_dataset(DATASET_DIR)

    samples_out = []
    latencies = []

    # =========================
    # INFERENCE
    # =========================
    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Running inference"):
        batch = dataset[start:start + args.batch_size]
        messages = []
        prompts_used = []

        for ex in batch:
            prompt = ex["meta"].get("prompt", "")
            prompt = re.sub(r'(\[)\s*y\s*,\s*x\s*(\])', r'\1x, y\2', prompt, flags=re.IGNORECASE)
            prompt = f"# USER REQUEST #: {prompt}\n"

            if args.prompt_position == "prepend":
                prompt = base_prompt + "\n\n" + prompt
            elif args.prompt_position == "append":
                prompt = prompt + "\n\n" + base_prompt

            prompts_used.append(prompt)

            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image"]},
                        {"type": "text", "text": prompt}
                    ]
                }
            ])

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True
        ).to(model.device)

        start_t = time.time()
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty
        )
        batch_time = time.time() - start_t

        outputs = [
            g[len(i):] for i, g in zip(inputs.input_ids, generated)
        ]

        decoded = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        for i, text in enumerate(decoded):
            samples_out.append({
                "index": start + i,
                "sample_id": batch[i]["id"],
                "prompt": prompts_used[i],
                "output": text,
                "non_violating_objects_bounding_boxes": batch[i]["meta"].get(
                    "non_violating_objects_bounding_boxes", []
                ),
                "violating_objects_bounding_boxes": batch[i]["meta"].get(
                    "violating_objects_bounding_boxes", []
                ),
                "latency_sec": batch_time / len(batch)
            })
            latencies.append(batch_time / len(batch))

        del inputs, generated, outputs, decoded
        torch.cuda.empty_cache()
        gc.collect()

    # =========================
    # WRITE OUTPUT
    # =========================
    output_json = {
        "metadata": {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "prompt_version": args.prompt_version,
            "prompt_position": args.prompt_position,
            "exact_prompt_text": base_prompt,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "generation_params": {
                "top_p": args.top_p,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty
            },
            "dataset_dir": DATASET_DIR,
            "num_samples": len(dataset),
            "run_timestamp": run_time,
            "avg_latency_sec": sum(latencies) / len(latencies)
        },
        "samples": samples_out
    }

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"[INFO] Saved results to {output_path}")


if __name__ == "__main__":
    main()
