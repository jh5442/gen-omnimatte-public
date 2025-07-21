import os
import glob
import json
import argparse
import cv2
import uuid
import gc
import numpy as np
from PIL import Image
import imageio.v3 as iio
import mediapy as media
import gradio as gr
from loguru import logger
import torch
import decord
from matplotlib import colormaps

from inference.cogvideox_fun.predict_v2v import load_pipeline
from videox_fun.utils.utils import get_video_mask_input, save_videos_grid, save_inout_row
from config.default_cogvideox import get_config as get_config
from config.default_omnimatte import get_config as get_config_omnimatte

from omnimatte.utils import VideoMaskTracker, transfer_detail
from omnimatte.animation import read_layer_from_frames, visualize_layers
from inference.reconstruct_omnimatte import run_optimization

parser = argparse.ArgumentParser(description="Gradio demo for Generative Omnimatte")
parser.add_argument(
    "--transformer_path",
    type=str,
    default="PATH/TO/COGVIDEOX/CASPER/TRANSFORMER/diffusion_pytorch_model.safetensors",
)
args = parser.parse_args()

TMP_DIR = "/home/ubuntu/jin-Vol/results/demos_for_product/gen_omnimatte/mt_lab_videos/test_06"
SEQ_NAME = "removal"
CONFIG = get_config()
CONFIG.video_model.transformer_path = args.transformer_path
MAX_LENGTH = CONFIG.data.max_video_length
SAMPLE_SIZE = tuple(map(int, CONFIG.data.sample_size.split('x')))  # (H, W)
FPS = CONFIG.data.fps
MAX_DEMO_OBJS = 3
ORIGINAL_VIDEO_LEN = 0

CONFIG_OMNIMATTE = get_config_omnimatte()

SAM_TRACKER = VideoMaskTracker()


def preprocess_video(video_in):
    global ORIGINAL_VIDEO_LEN, SEQ_NAME
    logger.debug(f"Preprocessing video: {video_in}")
    SEQ_NAME = "gradio_demo" + '-' + str(uuid.uuid4())[:8]
    seq_dir = os.path.join(TMP_DIR, SEQ_NAME)
    if os.path.exists(seq_dir):
        logger.debug(f"Removing existing directory: {seq_dir}")
        os.system(f"rm -rf {seq_dir}")
    os.makedirs(seq_dir, exist_ok=True)

    output_path = os.path.join(seq_dir, "input_video.mp4")
    with media.VideoReader(video_in) as r:
        with media.VideoWriter(output_path, shape=SAMPLE_SIZE, fps=FPS) as out:
            for i, frame in enumerate(r):
                if i >= MAX_LENGTH:
                    break
                frame_res = Image.fromarray(frame).resize((SAMPLE_SIZE[1], SAMPLE_SIZE[0]), resample=Image.LANCZOS)
                out.add_image(np.array(frame_res))

    logger.debug(f"Preprocessed video saved to {output_path}")
    ORIGINAL_VIDEO_LEN = i + 1
    logger.debug(f"Total frames (in process_video): {ORIGINAL_VIDEO_LEN}")

    return output_path


def load_ref_image(video_preproc, ref_index_norm):
    vr = decord.VideoReader(video_preproc, ctx=decord.cpu(0))
    video_len = len(vr)
    ref_index = int(ref_index_norm * (video_len - 1))

    image_extracted = vr[ref_index].asnumpy()
    del vr
    return image_extracted, image_extracted, ref_index


def prepare_trimask(sam_masks):
    logger.debug("Converting to trimask...")

    seq_dir = os.path.join(TMP_DIR, SEQ_NAME)
    num_fgs = sam_masks.max()
    assert num_fgs <= MAX_DEMO_OBJS
    for i in range(num_fgs):
        mask_i = (sam_masks == (i + 1)).astype(float)
        media.write_video(os.path.join(seq_dir, f'mask_{i:02d}.mp4'), mask_i, fps=FPS)

    fg_ids = [-1]
    if num_fgs > 1:
        fg_ids.extend(list(range(num_fgs)))

    input_masks = []
    for i, fg_id in enumerate(fg_ids):
        logger.debug(f"Processing foreground {fg_id}...")
        input_video, input_video_mask, prompt, _ = get_video_mask_input(
            SEQ_NAME,
            data_rootdir=TMP_DIR,
            keep_fg_ids=[fg_id],
            use_trimask=CONFIG.video_model.use_trimask,
            dilate_width=CONFIG.data.dilate_width,
            max_video_length=MAX_LENGTH,
            sample_size=SAMPLE_SIZE,
        )
        input_masks.append(input_video_mask)

    input_masks_viz = 1 - torch.cat(input_masks, dim=-2)[0, 0].cpu().numpy()
    viz_mask_path = os.path.join(seq_dir, "viz_trimasks.mp4")
    media.write_video(viz_mask_path, input_masks_viz, fps=FPS)

    return viz_mask_path, [input_video, input_masks, prompt]


def edit_trimask(selected_points, input_pkg):
    logger.debug("Editing trimask...")
    seq_dir = os.path.join(TMP_DIR, SEQ_NAME)
    input_video, input_masks, prompt = input_pkg

    _, _, t, h, w = input_video.shape

    for i in range(len(input_masks)):
        if i >= len(selected_points): break
        selected_points_i = selected_points[i]
        input_mask_ref = input_masks[i][:, :, :1, :, :]
        for (x, y), label in selected_points_i:
            ksize = 5
            x, y = int(x), int(y)
            x_min = max(0, x - ksize)
            x_max = min(w, x + ksize)
            y_min = max(0, y - ksize)
            y_max = min(h, y + ksize)
            input_mask_ref[:, :, :, y_min:y_max, x_min:x_max] = 1. - label
        input_masks[i][:, :, :1, :, :] = input_mask_ref

    input_pkg = [input_video, input_masks, prompt]

    input_masks_viz = 1 - torch.cat(input_masks, dim=-2)[0, 0].cpu().numpy()
    viz_mask_path = os.path.join(seq_dir, f"viz_trimasks_edited_{str(uuid.uuid4())[:8]}.mp4")
    media.write_video(viz_mask_path, input_masks_viz, fps=FPS)

    return viz_mask_path, input_pkg


def stack_videos(video_paths, save_path, stack_dim=1):
    video_list = [media.read_video(video_path) for video_path in video_paths]
    video_list_border = [video_list[0]]
    t, h, w = video_list[0].shape[:3]
    for video in video_list[1:]:
        assert video.shape[0] == t and video.shape[1] == h and video.shape[2] == w
        if stack_dim == 1:
            border = np.ones((t, 5, w, 3), dtype=np.uint8) * 255
        elif stack_dim == 2:
            border = np.ones((t, h, 5, 3), dtype=np.uint8) * 255
        else:
            raise ValueError(f"Invalid stack_dim: {stack_dim}. Must be 1 or 2.")
        video_list_border.append(border)
        video_list_border.append(video)

    video_stacked = np.concatenate(video_list_border, axis=stack_dim)
    media.write_video(save_path, video_stacked, fps=FPS)


# user click the image to get points, and show the points on the image
def get_point(img, sel_pix, obj_id, point_type, evt: gr.SelectData):
    obj_id = int(obj_id)

    while obj_id >= len(sel_pix):
        sel_pix.append([])
    if point_type == 'positive':
        sel_pix[obj_id].append((np.array(evt.index), 1))
    elif point_type == 'negative':
        sel_pix[obj_id].append((np.array(evt.index), 0))
    else:
        sel_pix[obj_id].append((np.array(evt.index), 1))

    # draw points
    for _obj_id in range(len(sel_pix)):
        for point, label in sel_pix[_obj_id]:
            color = get_palette()[_obj_id*3+3:_obj_id*3+6]
            mark = 1 if label == 0 else 5
            cv2.drawMarker(img, point, color, markerType=mark, markerSize=10, thickness=5)

    return img if isinstance(img, np.ndarray) else np.array(img)


def undo_points(image_backup, selected_points):
    while len(selected_points):
        selected_points.pop()
    return image_backup if isinstance(image_backup, np.ndarray) else np.array(image_backup)


def process_selected_points(selected_points):
    max_num_pts = 0
    selected_points_new = []
    labels_new = []
    for pts_i in selected_points:
        max_num_pts = max(max_num_pts, len(pts_i))
    for pts_i in selected_points:
        pts_i_new, labels_i = [], []
        for i in range(max_num_pts):
            pts_i_new.append(pts_i[i % len(pts_i)][0])
            labels_i.append(pts_i[i % len(pts_i)][1])

        selected_points_new.append(np.array(pts_i_new).reshape(-1, 2))
        labels_new.append(np.array(labels_i).reshape(-1))
    selected_points_new = np.stack(selected_points_new, axis=0)
    labels_new = np.stack(labels_new, axis=0)
    return selected_points_new, labels_new


def convert_video_to_frame_dir(video):
    tmp_dir = os.path.join(TMP_DIR, f'mask-frames-{str(uuid.uuid4())}')
    os.makedirs(tmp_dir, exist_ok=True)
    if isinstance(video, str) and video.endswith('.mp4'):
        video = media.read_video(video)

    for i, image in enumerate(video):
        media.write_image(os.path.join(tmp_dir, f'{i:05d}.jpg'), image)

    return tmp_dir


def get_palette():
    palette = []
    cmap = colormaps['Set1']
    for i in range(MAX_DEMO_OBJS + 1):
        rgb = np.uint8(np.array(cmap(i))[:3] * 255)
        palette.extend(rgb.tolist())

    return palette


def visualize_image_segmentation(image, mask):
    pil_mask = Image.fromarray(mask.astype(np.uint8))
    pil_mask = pil_mask.convert('P')
    pil_mask.putpalette(get_palette())

    image = Image.fromarray(image)
    alpha = 127
    image.paste(pil_mask, (0, 0), Image.fromarray(np.uint8((mask > 0) * alpha)))
    return np.array(image)


def segment_image(image, selected_points):
    selected_points, labels = process_selected_points(selected_points)
    logger.debug(f"Selected points: {selected_points}")
    logger.debug(f"Labels: {labels}")
    tmp_dir = convert_video_to_frame_dir(image[None])
    mask = SAM_TRACKER.run(tmp_dir, selected_points, 0, labels=labels)[0]
    os.system(f"rm -r {tmp_dir}")
    viz = visualize_image_segmentation(image, mask)
    media.write_image(os.path.join(TMP_DIR, SEQ_NAME, "image_mask_viz.png"), viz)
    return viz, mask


def segment_video(video, ref_idx, selected_points):
    video = media.read_video(video)
    selected_points, labels = process_selected_points(selected_points)
    tmp_dir = convert_video_to_frame_dir(video)
    masks = SAM_TRACKER.run(tmp_dir, selected_points, ref_idx, labels=labels)
    os.system(f"rm -r {tmp_dir}")
    output_path = os.path.join(TMP_DIR, SEQ_NAME, "video_mask_viz.mp4")
    viz = []
    for frame, frame_mask in zip(video, masks):
        viz.append(visualize_image_segmentation(frame, frame_mask))
    media.write_video(output_path, np.stack(viz), fps=FPS)
    return output_path, masks


def run_casper(sam_masks, prompt, num_sampling_steps):
    global ORIGINAL_VIDEO_LEN
    logger.debug(f"Total frames in run_casper: {ORIGINAL_VIDEO_LEN}")

    pipeline, vae, generator = load_pipeline(CONFIG)

    logger.debug("Running Casper...")
    seq_dir = os.path.join(TMP_DIR, SEQ_NAME)
    save_dir = os.path.join(seq_dir, "casper_outputs")
    os.makedirs(save_dir, exist_ok=True)

    json.dump({"bg": prompt}, open(os.path.join(seq_dir, "prompt.json"), "w"))

    seq_dir = os.path.join(TMP_DIR, SEQ_NAME)
    num_fgs = sam_masks.max()
    assert num_fgs <= MAX_DEMO_OBJS
    for i in range(num_fgs):
        mask_i = (sam_masks == (i + 1)).astype(float)
        media.write_video(os.path.join(seq_dir, f'mask_{i:02d}.mp4'), mask_i, fps=FPS)

    fg_ids = [-1]
    if num_fgs > 1:
        fg_ids.extend(list(range(num_fgs)))

    all_save_paths = []
    for i, fg_id in enumerate(fg_ids):
        logger.debug(f"Processing foreground {fg_id}...")
        input_video, mask_i, prompt, _ = get_video_mask_input(
            SEQ_NAME,
            data_rootdir=TMP_DIR,
            keep_fg_ids=[fg_id],
            use_trimask=CONFIG.video_model.use_trimask,
            dilate_width=CONFIG.data.dilate_width,
            max_video_length=MAX_LENGTH,
            sample_size=SAMPLE_SIZE,
        )
        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=CONFIG.video_model.temporal_window_size,
                negative_prompt=CONFIG.video_model.negative_prompt,
                height=SAMPLE_SIZE[0],
                width=SAMPLE_SIZE[1],
                generator=generator,
                num_inference_steps=int(num_sampling_steps),
                guidance_scale=CONFIG.video_model.guidance_scale,
                video=input_video,
                mask_video=mask_i,
                strength =CONFIG.video_model.denoise_strength,
                use_trimask=CONFIG.video_model.use_trimask,
                zero_out_mask_region=CONFIG.video_model.zero_out_mask_region,
                use_vae_mask=CONFIG.video_model.use_vae_mask,
                stack_mask=CONFIG.video_model.stack_mask,
            ).videos  # (1, 3, T, H, W)

            logger.debug(f"Sample shape: {sample.shape}")
            logger.debug(f"Input video shape: {input_video.shape}")
            logger.debug(f"Input mask shape: {mask_i.shape}")
            logger.debug(f"ORIGINAL_VIDEO_LEN: {ORIGINAL_VIDEO_LEN}")

            sample = sample[:, :, :ORIGINAL_VIDEO_LEN, :, :]
            logger.debug(f"Sample shape: {sample.shape}")
            save_path = os.path.join(save_dir, f"{SEQ_NAME}-fg={fg_id:02d}-0001.mp4")
            save_videos_grid(sample, save_path, fps=FPS)
            row_path = save_path[:-4] + "_tuple.mp4"

            print(input_video.shape, mask_i.shape, sample.shape)

            save_inout_row(
                input_video[:, :, :ORIGINAL_VIDEO_LEN, :, :],
                mask_i[:, :, :ORIGINAL_VIDEO_LEN, :, :],
                sample,
                row_path,
                fps=FPS,
                visualize_error=False,
            )

        all_save_paths.append(row_path)

    del pipeline
    del vae
    del generator
    torch.cuda.empty_cache()
    gc.collect()

    if len(all_save_paths) > 1:
        save_path = os.path.join(seq_dir, "stacked_casper_outputs.mp4")
        stack_videos(all_save_paths, save_path=save_path, stack_dim=1)
        return save_path
    else:
        return all_save_paths[0]


def run_omnimatte():
    config = CONFIG_OMNIMATTE
    config.data.data_rootdir = TMP_DIR
    seq_dir = os.path.join(TMP_DIR, SEQ_NAME)
    config.omnimatte.source_video_dir = os.path.join(seq_dir, "casper_outputs")
    config.experiment.save_path = os.path.join(seq_dir, "omnimatte_outputs")
    config.data.skip_if_exists = True
    config.omnimatte.num_steps = 4000  # for faster demo, reduce the number of steps

    num_fgs = len(glob.glob(os.path.join(seq_dir, 'mask_*.mp4')))
    if num_fgs == 1:
        fg_ids = [-1]
    else:
        fg_ids = list(range(num_fgs))

    for fg_id in fg_ids:
        logger.info(f"Running Omnimatte for fg_id: {fg_id}")
        run_optimization(config, SEQ_NAME, fg_id, num_fgs)

    omnimatte_output_dir = os.path.join(config.experiment.save_path, SEQ_NAME)
    if config.omnimatte.detail_transfer:
        transfer_detail(config, SEQ_NAME, num_fgs)

        layer_paths = sorted(glob.glob(os.path.join(omnimatte_output_dir, "dt_fg*_rgba_checker.mp4")))
    else:
        layer_paths = sorted(glob.glob(os.path.join(omnimatte_output_dir, "fg*_rgba_checker.mp4")))

    layer_paths = [os.path.join(omnimatte_output_dir, "bg.mp4")] + layer_paths
    omnimatte_vis_path = os.path.join(seq_dir, "stacked_omnimatte_outputs.mp4")
    stack_videos(layer_paths, save_path=omnimatte_vis_path, stack_dim=2)

    layer_vis_path = os.path.join(seq_dir, "layer_visualization.mp4")

    vid_in = media.read_video(os.path.join(seq_dir, "input_video.mp4")).astype(float) / 255.
    layers = []
    for layer_path in layer_paths[1:]:  # fgs
        layers.append(read_layer_from_frames(layer_path.replace('_rgba_checker.mp4', '')))
    layers.append(read_layer_from_frames(layer_paths[0].replace('.mp4', '')))  # bg the last
    layers = np.stack(layers, axis=0)
    visualize_layers(vid_in, layers, save_path=layer_vis_path, fps=FPS)

    return omnimatte_vis_path, layer_vis_path


def save_all(save_path):
    seq_dir = os.path.join(TMP_DIR, SEQ_NAME)
    logger.debug(f"Copying files from {seq_dir} to {save_path}")
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    os.system(f"cp -r {seq_dir} {save_path}")
    return "Saved!"


if __name__ == '__main__':
    os.makedirs(TMP_DIR, exist_ok=True)

    with gr.Blocks() as demo:
        gr.Markdown('# Generative Omnimatte Public Reimplementation')

        ### 1. Loading input video and reference image for segmentation
        gr.Markdown('<hr>')
        gr.Markdown('## 1. Upload video')
        gr.Markdown('### Steps: Upload video → Slide to a desired timestamp to extract reference image → Load reference image')
        gr.Markdown(f"### Note: the model can only handles the first {MAX_LENGTH} "
                    f"frames and will be resized to HxW={SAMPLE_SIZE} for inference.")
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Video Input", format="mp4")

                boys_beach = os.path.join(os.path.dirname(__file__), "examples", "boys-beach", "input_video.mp4")
                animator_draw = os.path.join(os.path.dirname(__file__), "examples", "animator-draw", "input_video.mp4")
                boat_shore = os.path.join(os.path.dirname(__file__), "examples", "boat-shore", "input_video.mp4")

                gr.Examples(examples=[boys_beach, animator_draw, boat_shore],
                            inputs = [video_in])

            with gr.Column():
                video_preproc = gr.Video(label="Preprocessed Video", format="mp4")

                ref_index_norm = gr.Slider(label="Reference Frame Index", value=0, minimum=0, maximum=1.0, step=0.01)
                ref_index_int = gr.State(value=0)
        with gr.Row():
            btn_upload = gr.Button("Upload Video")
            btn_load_ref = gr.Button("Load Reference Image for Segmentation")

        ### 2. Selecting foreground object masks
        gr.Markdown('<hr>')
        gr.Markdown('## 2. Obtain video masks using SAM2')
        gr.Markdown(f"### Steps: Select Points → Segment Image → Segment Video → Type prompt → Run Casper")
        gr.Markdown(f"### Note: you can select up to {MAX_DEMO_OBJS} objects in this demo.")
        gr.Markdown('- Always select FG ID starting from 0.')
        gr.Markdown('- (Optional) Specify negative points can avoid false-positive segmentation in SAM2. Default: positive')
        gr.Markdown('- (Optional) Segment the reference image to quickly check if the masks are correct before video segmentation.')
        with gr.Row():
            with gr.Column():
                image_backup = gr.State(value=None)
                image_mask = gr.State(value=None)
                video_masks = gr.State(value=None)
                with gr.Tab(label="Point selection"):
                    image_ref = gr.Image(label="Reference Image", type="numpy")

                with gr.Row():
                    selected_points = gr.State([])      # store points
                    fg_id = gr.Number(value=0, minimum=0, maximum=MAX_DEMO_OBJS - 1, label="FG ID")
                    radio = gr.Radio(['positive', 'negative'], label='Point labels')

                with gr.Row():
                    btn_undo_select = gr.Button('Undo selected points')
                    btn_segment_image = gr.Button("Segment Reference Image")
                    btn_segment_video = gr.Button("Segment Video")

            with gr.Column():
                with gr.Tab(label="Image Segmentation"):
                    image_seg = gr.Image(label="Image Segmentation", type="numpy")
                with gr.Tab(label="Video Segmentation"):
                    video_seg = gr.Video()

        gr.Markdown('<hr>')
        gr.Markdown('## 3. Run Casper (Object Effect Removal)')
        gr.Markdown('- Takes a few minutess to run')
        with gr.Column():
            with gr.Row():
                tb_prompt = gr.Textbox(
                    label="Prompt (simply describing the target clean background)",
                    placeholder="a beautiful empty background scene."
                )
                num_sample_steps = gr.Slider(value=4, minimum=1, maximum=50, step=1, label="Number of sampling steps")
            btn_run_casper = gr.Button("Run Casper")
            with gr.Row():
                gr.Markdown('Input video')
                gr.Markdown('Input triamsk')
                gr.Markdown('Output video')
            video_casper_out = gr.Video(label="Casper Outputs", format="mp4")

        gr.Markdown('<hr>')
        gr.Markdown('## 4. Omnimatte Optimization')
        gr.Markdown('- Takes ~8 minutes per layer')
        with gr.Column():
            btn_run_omnimatte = gr.Button("Run Omnimatte Optimization")
            video_omnimatte_out = gr.Video(label="Omnimatte Outputs", format="mp4")
            video_animation = gr.Video(label="Layer Visualization", format="mp4")

            gr.Markdown('### Optional: save the results')
            tb_save_path = gr.Textbox(label='Save Path', placeholder='Path to save the results')
            btn_save = gr.Button("Save")
            tb_save_status = gr.Textbox(label="Saving status", value="Saving...")

        # buttons
        btn_upload.click(
            fn=preprocess_video, inputs=[video_in], outputs=[video_preproc])
        btn_load_ref.click(
            fn=load_ref_image, inputs=[video_preproc, ref_index_norm], outputs=[image_ref, image_backup, ref_index_int])
        image_ref.select(
            fn=get_point, inputs=[image_ref, selected_points, fg_id, radio], outputs=[image_ref])
        btn_undo_select.click(
            fn=undo_points, inputs=[image_backup, selected_points], outputs=[image_ref])
        btn_segment_image.click(
            fn=segment_image, inputs=[image_backup, selected_points], outputs=[image_seg, image_mask])
        btn_segment_video.click(
            fn=segment_video, inputs=[video_preproc, ref_index_int, selected_points], outputs=[video_seg, video_masks])
        btn_run_casper.click(
            fn=run_casper, inputs=[video_masks, tb_prompt, num_sample_steps], outputs=[video_casper_out])
        btn_run_omnimatte.click(
            fn=run_omnimatte, inputs=[], outputs=[video_omnimatte_out, video_animation])
        btn_save.click(
            fn=save_all, inputs=[tb_save_path], outputs=[tb_save_status])

    demo.queue().launch(debug=True, share=True)
