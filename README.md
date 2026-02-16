# SafeCoVT: Safe Chain-of-Visual-Thought

Official implementation of **SafeCoVT: A Plug-and-Play Framework for Safety-Critical Applications**.

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](link-to-paper)
[![Dataset](https://img.shields.io/badge/Dataset-ASIMOV--2.0-blue)](https://asimov-benchmark.github.io/v2/)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Experiments](#reproducing-experiments)
  - [Text-Only Baselines](#text-only-baselines)
  - [SafeCoVT](#safecovt)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+

### 1. Clone Repository
```bash
git clone https://github.com/Navlika-Singh/SafeCoVT.git
cd SafeCoVT
```

### 2. Setup Environments

We provide two conda environments for different components:

**a) Text-Only Baselines & SafeCoVT:**
```bash
conda env create -f env/safecovt_environment.yml
conda activate safecovt
```

**b) Vision Tools (SAM, Grounding-DINO, Depth Anything):**
```bash
conda env create -f env/sketchpad_environment.yml
conda activate sketchpad
```
Follow detailed tool installation: [VisualSketchpad Installation Guide](https://github.com/Yushi-Hu/VisualSketchpad/blob/main/vision_experts/installation.md)

### 3. Download Pre-trained Models

**Qwen3-VL Models** (for baselines and SafeCoVT):
```bash
# Using Hugging Face CLI
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ./models/Qwen3-VL-2B
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ./models/Qwen3-VL-4B
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/Qwen3-VL-8B
```

**Vision Tools** (SAM, Grounding-DINO, Depth Anything):
 Follow setup instructions in `VisualSketchpad/vision_experts/installation.md`
### 4. Dataset
The ASIMOV-2.0 dataset (both original and clean versions) are available at:
```
data/
├── asimov_v2_constraints_with_rationale/  # Original
└── cleaned/                                # Corrected
```

---

## Quick Start

### Running a Single Experiment

**Text-Only CoT (4B model):**
```bash
conda activate safecovt
python scripts/main.py \
  --model_path ./models/Qwen3-VL-4B \
  --model_name Qwen3VL_4B \
  --prompt_version cot \
  --prompt_position prepend \
  --batch_size 4 \
  --output_dir ./results/baseline_cot
```

**SafeCoVT (4B model):**
```bash
# First, start vision tool servers (in separate terminals with sketchpad env)
conda activate sketchpad
cd VisualSketchpad/vision_experts/Depth-Anything
python depthanything_server.py  # Copy the Gradio link

cd ../simplified_som
python som_server.py  # Copy the Gradio link

cd ../GroundingDINO
python grounding_dino_server.py  # Copy the Gradio link

# Update config with server URLs
# Edit VisualSketchpad/agent/config.py:
#   DEPTH_ANYTHING_ADDRESS = "http://127.0.0.1:XXXX"
#   SOM_ADDRESS = "http://127.0.0.1:YYYY"
#   GROUNDING_DINO_ADDRESS = "http://127.0.0.1:ZZZZ"

# Run SafeCoVT
conda activate safecovt
cd VisualSketchpad/agent
bash SafeCoVT.sh
```

---

## Project Structure
```
SafeCoVT/
├── scripts/
│   ├── main.py              # Text-only baseline inference
│   └── prompts.json         # Prompts for all baselines
├── VisualSketchpad/
│   ├── agent/
│   │   ├── safecovt_main.py # SafeCoVT inference
│   │   ├── tools.py         # Tool implementations
│   │   ├── config.py        # Tool server addresses
│   │   └── SafeCoVT.sh      # Run script
│   └── vision_experts/      # Tool servers (SAM, DINO, Depth)
├── bashScripts/
│   └── test.pbs             # HPC job script
├── env/
│   ├── safecovt_environment.yml
│   ├── sketchpad_environment.yml
│   └── covt_environment.yml
├── data/
│   ├── asimov_v2_constraints_with_rationale/
│   └── cleaned/
└── README.md
```

---

## Acknowledgments

This work builds upon:
- **Visual Sketchpad** for tool-augmented reasoning: [GitHub](https://github.com/Yushi-Hu/VisualSketchpad)
- **Qwen3-VL** for multimodal language models: [GitHub](https://github.com/QwenLM/Qwen3-VL)
- **CoVT** for visual token generation: [GitHub](https://github.com/Wakals/CoVT)
- **ASIMOV-2.0** for safety benchmarking: [Website](https://asimov-benchmark.github.io/v2/)

We thank Dr. Oana-Maria Camburu for supervision and Imperial College London HPC for computational resources.

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Navlika Singh: [navlika.singh24@imperial.ac.uk](mailto:navlika.singh24@imperial.ac.uk)

---

##  License

This project is licensed under the MIT License - see LICENSE file for details.