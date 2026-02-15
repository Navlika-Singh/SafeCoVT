from main import run_agent
import os, glob, argparse
from tqdm import tqdm

import os
import glob
from tqdm import tqdm

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_PATH = "/rds/general/user/ns1324/home/iso/Qwen3-VL/pretrainedModels/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"

def run_task(task, output_dir, task_type="vision", task_name=None):
    all_task_instances = glob.glob(
        f"/rds/general/user/ns1324/home/iso/VisualSketchpad/tasks/{task}/processed/*/"
        if task_type == "vision"
        else f"../tasks/{task}/*/"
    )

    output_dir = os.path.join(output_dir, task)
    os.makedirs(output_dir, exist_ok=True)

    # Load local Qwen model & processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {DEFAULT_MODEL_PATH}...")
    model = AutoModelForImageTextToText.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        repetition_penalty=1.0
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH)
    print(f"Model loaded on {device}")

    for task_instance in tqdm(all_task_instances):
        instance_name = os.path.basename(os.path.normpath(task_instance))
        instance_output_dir = os.path.join(output_dir, instance_name)

        # ✅ SKIP if already exists
        # if os.path.exists(instance_output_dir):
        #     print(f"Skipping (already exists): {instance_name}")
        #     continue

        print(f"Running task instance: {task_instance}")
        run_agent(
            model,
            processor,
            task_instance,
            output_dir,
            task_type=task_type,
            task_name=task_name,
        )
        exit()

        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["asimov_constraint", "vstar", "blink_viscorr", "blink_semcorr", "blink_depth",
                                                    "blink_jigsaw", "blink_spatial", "mmvp", 
                                                    "geometry", 
                                                    "graph_connectivity", "graph_isomorphism", "graph_maxflow", 
                                                    "math_convexity", "math_parity", "winner_id"], help="The task name")
    args = parser.parse_args()
    
    if args.task in ["asimov_constraint", "vstar", "blink_viscorr", "blink_semcorr", "blink_depth","blink_jigsaw", "blink_spatial", "mmvp",]:
        task_type = "vision"
        task_name = None
        
    elif args.task in ["geometry"]:
        task_type = "geo"
        task_name = None
        
    else:
        task_type = "math"
        task_name = args.task
        
    run_task(args.task, "outputs", task_type=task_type, task_name=task_name)
