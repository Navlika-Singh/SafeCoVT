# Prepare anchor models

## SAM

SAM is the anchor model for segmentation model. To prepare its checkpoint, download the [VITiT-H SAM Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) to the `./src/anchors/segment_anything/ckpt/`.


## DepthAnything v2

DepthAnthing v2 is the anchor model for depth estimation. Download [DepthAnthing v2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) to the `./src/anchors/DepthAnything/ckpt`.

## PIDINet

PIDINet is for edge detection. It is very lightweight, to download the ckpt file, view [PIDINet checkpoint](https://github.com/hellozhuo/pidinet/blob/master/trained_models/table5_baseline.pth), download it and put it under `./src/anchors/pidinet/ckpt`.
