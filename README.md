# Generative Omnimatte: Learning to Decompose Video into Layers (CVPR 2025 Highlight)

<div style="line-height: 1;">
  <a href="https://gen-omnimatte.github.io/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website-Gen--Omnimatte-4285F4" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2411.16683" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Gen--Omnimatte-FBBC06" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://www.youtube.com/watch?v=SD-VCNvTBg4" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/badge/YouTube-Gen--Omnimatte-EA4335" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<h4>

[Yao-Chih Lee](https://yaochih.github.io)<sup>1,2</sup>,
[Erika Lu](https://erikalu.com/)<sup>1</sup>,
[Sarah Rumbley](https://scholar.google.com/citations?user=gPkCTQ0AAAAJ&hl=en)<sup>1</sup>, 
[Michal Geyer](https://michalgeyer.my.canva.site/)<sup>1,3</sup>,
[Jia-Bin Huang](https://jbhuang0604.github.io/)<sup>2</sup>,
[Tali Dekel](https://www.weizmann.ac.il/math/dekel/home)<sup>1,3</sup>,
[Forrester Cole](https://people.csail.mit.edu/fcole/)<sup>1</sup>

<sup>1</sup>Google DeepMind, <sup>2</sup>University of Maryland, <sup>3</sup>Weizmann Institute of Science

</h4>

<hr>

<video src="https://github.com/user-attachments/assets/224f8e9f-f7d4-4236-8716-105d8187ae46" width="100%" controls autoplay loop muted></video>

## ❗ This is a **public reimplementation** of Generative Omnimatte

We applied the same fine-tuning strategy used for the original Casper model (video object-effect removal) to public video diffusion models, [CogVideoX](https://github.com/THUDM/CogVideo) and [Wan2.1](https://github.com/Wan-Video/Wan2.1), with minimum modifications. However, ***the performance of these fine-tuned public models is close to, but does not match that of the Lumiere-based Casper***. We hope continued development will lead to future performance improvements.

This public reimplementation builds on code and models from [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main). We thank the authors for sharing the codes and pretrained inpainting models for CogVideoX and Wan2.1.

## Table of Contents
- [Environment](#env)
- [Casper (video object effect removal)](#casper-inference)
  - [Casper model weights](#model-weights)
  - [Casper inference](#casper-run)
  - [Training data](./datasets/README.md)
  - [Casper training](#casper-training)
- [Omnimatte optimization](#omnimatte)
- [Gradio Demo](#gradio)
- [Acknowledgments](#ack)
- [Citation](#cite)
- [License](#license)

## Environment <a name="env"></a>
- Tested on python 3.10, CUDA 12.4, torch 2.5.1, diffusers 0.32.2
- Please check [requirements.txt](./requirements.txt) for the dependencies
- Install SAM2 by following the [instructions](https://github.com/facebookresearch/sam2/tree/main?tab=readme-ov-file#installation).

## Casper (Video Object Effect Removal) <a name="casper-inference"></a>

### Model Weights  <a name="model-weights"></a>
We provide several variants based on different model backbones. In additional to downloading our Casper model weight, please also download the pretrained inpainting model from [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main).

| Pretrained inpainting from VideoX-Fun | Casper model | Description |
| ------------- | ------------- | ------------- |
| [CogVideoX-Fun-V1.5-5b-InP](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | [google drive](https://drive.google.com/file/d/1KTHGtlJPCDhQGRX5xbF3U63AyI_2T_cJ/view?usp=drive_link) | (Recommended) This model may perform better and faster than the Wan-based Casper models but still not as good as the Lumiere-based Casper. It was fully-finetuned from our inpainting model, which was initially fine-tuned from VideoX-Fun's released model. During inference, it processes a temporal window of 85 frames and can handle 197 frames using temporal multidiffusion. The default inference resolution is 384x672 (HxW).
| [Wan2.1-Fun-1.3B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) (V1.0) | [google drive](https://drive.google.com/file/d/1n3Sv4d0pbTjfa5UhypEhTaylrSy2X4C1/view?usp=drive_link) | The model was fully-finteuned from VideoX-Fun's released inpainting model. During inference, it processes a temporal window of 81 frames and can handle 197 frames using temporal multidiffusion. The default inference resolution is 480x832 (HxW). |
| [Wan2.1-Fun-14B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) (V1.0) | [google drive](https://drive.google.com/file/d/1ojrscv60h2Hx__tfLTuO680_QGYRpL-V/view?usp=drive_link) | Due to the large model size, we applied LoRA-based fine-tuning on the top of VideoX-Fun's released inpainting model. During inference, it processes a temporal window of 81 frames and can handle 197 frames using temporal multidiffusion. The default inference resolution is 480x832 (HxW). |


### Run Casper <a name="casper-run"></a>

- CogVideoX-based (recommended)
  - Takes 1-2 mins on an A100 GPU, and 4-5 mins on an A6000 GPU with 48GB RAM
  ```
  python inference/cogvideox_fun/predict_v2v.py \
      --config.data.data_rootdir="examples" \
      --config.experiment.run_seqs="boys-beach,animator-draw" \
      --config.experiment.save_path="CASPER/OUTPUT/DIR" \
      --config.video_model.model_name="PATH/TO/CogVideoX-Fun-V1.5-5b-InP" \
      --config.video_model.transformer_path="PATH/TO/CASPER/TRANSFORMER.safetensors"
  ```

- Wan2.1-1.3B-based
  - Takes ~10 mins on an A6000 GPU with 48GB RAM
  ```
  python inference/wan2.1_fun/predict_v2v.py \
      --config.data.data_rootdir="examples" \
      --config.experiment.run_seqs="boys-beach,animator-draw" \
      --config.experiment.save_path="CASPER/OUTPUT/DIR" \
      --config.video_model.model_name="PATH/TO/Wan2.1-Fun-1.3B-InP" \
      --config.video_model.transformer_path="PATH/TO/CASPER/TRANSFORMER.safetensors"
  ```

- Wan2.1-14B-based (LoRA)
  - Takes ~55 mins on an A6000 GPU with 64GM RAM 
  ```
  python inference/wan2.1_fun/predict_v2v.py \
      --config.data.data_rootdir="examples" \
      --config.experiment.run_seqs="boys-beach,animator-draw" \
      --config.experiment.save_path="CASPER/OUTPUT/DIR" \
      --config.video_model.model_name="PATH/TO/Wan2.1-Fun-14B-InP" \
      --config.video_model.lora_path="PATH/TO/CASPER/LORA.safetensors" \
      --config.video_model.lora_weight=1.0 \
      --config.system.gpu_memory_mode="sequential_cpu_offload"
   ```

- To run your own sequences, please follow the format in `examples/boys-beach` to provide your own input video, video masks, and text prompt in a folder. 
- Modify `--config.data.data_rootdir` and `--config.experiment.run_seqs` if needed.


### Casper training <a name="casper-training"></a>
- Please prepare the training data and the merged json file for the whole dataset by following the instructions in [./datasets](./datasets/README.md)
- Replace the absolute dataset path for `DATASET_META_NAME` before running the training scripts
  - CogVideoX Casper: `./scripts/cogvideox_fun/train_casper.sh`
  - Wan2.1-1.3B: `./scripts/wan2.1_fun/train_casper.sh`
  - Wan2.1-14B: `./scripts/wan2.1_fun/train_casper_lora.sh`
- We finetuned the public models on 4 H100 GPUs


## Omnimatte Optimization <a name="omnimatte"></a>
```
python inference/reconstruct_omnimatte.py \
  --config.data.data_rootdir="examples" \
  --config.experiment.run_seqs="boys-beach,animator-draw" \
  --config.omnimatte.source_video_dir="CASPER/OUTPUT/DIR" \
  --config.experiment.save_path="OMNIMATTE/OUTPUT/DIR" \
  --config.data.sample_size="384x672"
```
- Takes ~8 mins on an A6000 or A5000 GPU with 48GB RAM
- The argument `sample_size` should be the same resolution that you used to run the Casper model. (`384x672` for CogVideoX and `480x832` for Wan by default.)
- The results may be suboptimal as the optimization operates on output videos that include undesired artifacts due to the VAE of the latent-based DiT models.
- The current implementation processes a video of multiple objects sequentially rather than in parallel. In the future, speed can be improved by running individual omnimatte optimization in parallel if have multi GPUs.


## Gradio Demo <a name="gradio"></a>
```
GRADIO_TEMP_DIR=".tmp" python app.py \
  --transformer_path PATH/TO/COGVIDEOX/CASPER/diffusion_pytorch_model.safetensors
```
- Tested on an A6000 GPU with 48GB RAM
- The object-effect-removal step takes ~1 min per layer with 4 sampling steps for faster demo
- The omnimatte optimization step takes ~8 min per layer
![](./assets/gradio.png)

## Acknowledgments <a name="ack"></a>

We thank the authors of [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun), [SAM2](https://github.com/facebookresearch/sam2/tree/main), and [Omnimatte](https://github.com/erikalu/omnimatte) for their shared codes and models, and acknowledge the GPU resources provided by the [UMD HPC cluster](https://hpcc.umd.edu/hpcc/zaratan.html) for training our public video models.

We also appreciate the results from [Omnimatte](https://omnimatte.github.io/), [Omnimatte3D](https://openaccess.thecvf.com/content/CVPR2023/papers/Suhail_Omnimatte3D_Associating_Objects_and_Their_Effects_in_Unconstrained_Monocular_Video_CVPR_2023_paper.pdf), and [OmnimatteRF](https://omnimatte-rf.github.io/), as well as the videos on Pexels [[1](https://www.pexels.com/video/a-reflection-of-a-building-in-a-puddle-4935665/),[2](https://www.pexels.com/video/fahrrad-fahrt-durch-pfutze-27783285/),[3](https://www.pexels.com/video/racing-to-the-top-of-sand-dunes-855673/),[4](https://www.pexels.com/video/a-car-drifting-in-asphalt-road-4569087/),[5](https://www.pexels.com/video/a-car-drifting-on-a-racing-track-4568686/),[6](https://www.pexels.com/video/low-angle-shot-of-famous-brooklyn-bridge-5180403/),[7](https://www.pexels.com/video/car-driving-off-road-10358960/),[8](https://www.pexels.com/video/a-black-suv-driving-around-a-paved-road-in-a-desert-8642130/),[9](https://www.pexels.com/video/a-fast-car-on-the-road-5020089/),[10](https://www.pexels.com/video/a-boat-towing-a-small-boat-8212525/),[11](https://www.pexels.com/video/video-of-a-boat-in-the-middle-of-an-ocean-8703682/),[12](https://www.pexels.com/video/speed-boats-races-on-a-lake-2711117/),[13](https://www.pexels.com/video/a-fast-car-on-the-road-5020109/),[14](https://www.pexels.com/video/a-traveling-car-splashing-the-water-puddle-in-the-road-4309064/)], which were used for fine-tuning Casper.

## Citation <a name="cite"></a>
```
@inproceedings{generative-omnimatte,
  author    = {Lee, Yao-Chih and Lu, Erika and Rumbley, Sarah and Geyer, Michal and Huang, Jia-Bin and Dekel, Tali and Cole, Forrester},
  title     = {Generative Omnimatte: Learning to Decompose Video into Layers},
  booktitle = {CVPR},
  year      = {2025},
}
```


## License <a name="license"></a>
This project is licensed under [Apache-2.0 license](./LICENSE).

The CogVideoX-5B transformer is released under the [CogVideoX license](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
