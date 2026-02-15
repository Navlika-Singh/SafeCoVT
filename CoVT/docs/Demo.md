# CoVT VLM Gradio Demo

This project provides an interactive demo built with [Gradio](https://github.com/gradio-app/gradio), showcasing a conversational interface powered by the **CoVT Vision-Language Model (VLM)**.  
Users can upload images, ask questions, and chat with the model in real time through a simple and intuitive web UI.


## Installation

Make sure you have **Python 3.8+** installed. Use your current environment configured in the earlier [instructions](Eval.md), and run,
```bash
pip install gradio pillow
```

## Run the Gradio demo!

```bash
cd gradio
python gradio_demo.py

# or you can configure the GPUs you use by `CUDA_VISIBLE_DEVICES`
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py
```

Once the server starts, you will see an output similar to:
```nginx
Running on local URL:  http://127.0.0.1:7860
```
Open the link in your browser and you can interact with CoVT! 🚀

There is an example on the page of the demo, you can click it and run the inference. Also, you can replace `Wakals/CoVT-7B-seg_depth_dino` (in `gradio/gradio_demo.py` file) with other models in the [model zoo](https://huggingface.co/collections/Wakals/covt-chain-of-visual-thought).