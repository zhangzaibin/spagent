# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import os
import sys
import datetime
import imageio
import numpy as np
import torch
import gradio as gr

sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-3]))
from vace.models.ltx.ltx_vace import LTXVace


class FixedSizeQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []
    def add(self, item):
        self.queue.insert(0, item)
        if len(self.queue) > self.max_size:
            self.queue.pop()
    def get(self):
        return self.queue
    def __repr__(self):
        return str(self.queue)


class VACEInference:
    def __init__(self, cfg, skip_load=False, gallery_share=True, gallery_share_limit=5):
        self.cfg = cfg
        self.save_dir = cfg.save_dir
        self.gallery_share = gallery_share
        self.gallery_share_data = FixedSizeQueue(max_size=gallery_share_limit)
        if not skip_load:
            self.pipe = LTXVace(ckpt_path=args.ckpt_path,
                               text_encoder_path=args.text_encoder_path,
                               precision=args.precision,
                               stg_skip_layers=args.stg_skip_layers,
                               stg_mode=args.stg_mode,
                               offload_to_cpu=args.offload_to_cpu)

    def create_ui(self, *args, **kwargs):
        gr.Markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 15px;">
                        <a href="https://ali-vilab.github.io/VACE-Page/" style="text-decoration: none; color: inherit;">VACE-LTXV Demo</a>
                    </div>
                    """)
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1, min_width=0):
                self.src_video = gr.Video(
                    label="src_video",
                    sources=['upload'],
                    value=None,
                    interactive=True)
            with gr.Column(scale=1, min_width=0):
                self.src_mask = gr.Video(
                    label="src_mask",
                    sources=['upload'],
                    value=None,
                    interactive=True)
        #
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1, min_width=0):
                with gr.Row(equal_height=True):
                    self.src_ref_image_1 = gr.Image(label='src_ref_image_1',
                                                    height=200,
                                                    interactive=True,
                                                    type='filepath',
                                                    image_mode='RGB',
                                                    sources=['upload'],
                                                    elem_id="src_ref_image_1",
                                                    format='png')
                    self.src_ref_image_2 = gr.Image(label='src_ref_image_2',
                                                    height=200,
                                                    interactive=True,
                                                    type='filepath',
                                                    image_mode='RGB',
                                                    sources=['upload'],
                                                    elem_id="src_ref_image_2",
                                                    format='png')
                    self.src_ref_image_3 = gr.Image(label='src_ref_image_3',
                                                    height=200,
                                                    interactive=True,
                                                    type='filepath',
                                                    image_mode='RGB',
                                                    sources=['upload'],
                                                    elem_id="src_ref_image_3",
                                                    format='png')
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1):
                self.prompt = gr.Textbox(
                    show_label=False,
                    placeholder="positive_prompt_input",
                    elem_id='positive_prompt',
                    container=True,
                    autofocus=True,
                    elem_classes='type_row',
                    visible=True,
                    lines=2)
                self.negative_prompt = gr.Textbox(
                    show_label=False,
                    value="worst quality, inconsistent motion, blurry, jittery, distorted",
                    placeholder="negative_prompt_input",
                    elem_id='negative_prompt',
                    container=True,
                    autofocus=False,
                    elem_classes='type_row',
                    visible=True,
                    interactive=True,
                    lines=1)
        #
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1, min_width=0):
                with gr.Row(equal_height=True):
                    self.sample_steps = gr.Slider(
                        label='sample_steps',
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=40,
                        interactive=True)
                    self.context_scale = gr.Slider(
                        label='context_scale',
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        interactive=True)
                    self.guide_scale = gr.Slider(
                        label='guide_scale',
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        value=3.0,
                        interactive=True)
                    self.infer_seed = gr.Slider(minimum=-1,
                                                maximum=10000000,
                                                value=2025,
                                                label="Seed")
        #
        with gr.Accordion(label="Usable without source video", open=False):
            with gr.Row(equal_height=True):
                self.output_height = gr.Textbox(
                    label='resolutions_height',
                    value=512,
                    interactive=True)
                self.output_width = gr.Textbox(
                    label='resolutions_width',
                    value=768,
                    interactive=True)
                self.frame_rate = gr.Textbox(
                    label='frame_rate',
                    value=25,
                    interactive=True)
                self.num_frames = gr.Textbox(
                    label='num_frames',
                    value=97,
                    interactive=True)
        #
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                self.generate_button = gr.Button(
                    value='Run',
                    elem_classes='type_row',
                    elem_id='generate_button',
                    visible=True)
            with gr.Column(scale=1):
                self.refresh_button = gr.Button(value='\U0001f504')  # 🔄
        #
        self.output_gallery = gr.Gallery(
            label="output_gallery",
            value=[],
            interactive=False,
            allow_preview=True,
            preview=True)


    def generate(self, output_gallery, src_video, src_mask, src_ref_image_1, src_ref_image_2, src_ref_image_3, prompt, negative_prompt, sample_steps, context_scale, guide_scale, infer_seed, output_height, output_width, frame_rate, num_frames):

        output = self.pipe.generate(src_video=src_video,
                                    src_mask=src_mask,
                                    src_ref_images=[src_ref_image_1, src_ref_image_2, src_ref_image_3],
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    seed=infer_seed,
                                    num_inference_steps=sample_steps,
                                    num_images_per_prompt=1,
                                    context_scale=context_scale,
                                    guidance_scale=guide_scale,
                                    frame_rate=frame_rate,
                                    output_height=output_height,
                                    output_width=output_width,
                                    num_frames=num_frames)

        frame_rate = output['info']['frame_rate']
        name = '{0:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        video_path = os.path.join(self.save_dir, f'cur_gallery_{name}.mp4')
        video_frames = (torch.clamp(output['out_video'] / 2 + 0.5, min=0.0, max=1.0).permute(1, 2, 3, 0) * 255).cpu().numpy().astype(np.uint8)

        try:
            writer = imageio.get_writer(video_path, fps=frame_rate, codec='libx264', quality=8, macro_block_size=1)
            for frame in video_frames:
                writer.append_data(frame)
            writer.close()
            print(video_path)
        except Exception as e:
            raise gr.Error(f"Video save error: {e}")

        if self.gallery_share:
            self.gallery_share_data.add(video_path)
            return self.gallery_share_data.get()
        else:
            return [video_path]

    def set_callbacks(self, **kwargs):
        self.gen_inputs = [self.output_gallery, self.src_video, self.src_mask, self.src_ref_image_1, self.src_ref_image_2, self.src_ref_image_3, self.prompt, self.negative_prompt, self.sample_steps, self.context_scale, self.guide_scale, self.infer_seed, self.output_height, self.output_width, self.frame_rate, self.num_frames]
        self.gen_outputs = [self.output_gallery]
        self.generate_button.click(self.generate,
                                   inputs=self.gen_inputs,
                                   outputs=self.gen_outputs,
                                   queue=True)
        self.refresh_button.click(lambda x: self.gallery_share_data.get() if self.gallery_share else x, inputs=[self.output_gallery], outputs=[self.output_gallery])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for VACE-LTXV Demo:\n')
    parser.add_argument('--server_port', dest='server_port', help='', type=int, default=7860)
    parser.add_argument('--server_name', dest='server_name', help='', default='0.0.0.0')
    parser.add_argument('--root_path', dest='root_path', help='', default=None)
    parser.add_argument('--save_dir', dest='save_dir', help='', default='cache')
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='models/VACE-LTX-Video-0.9/ltx-video-2b-v0.9.safetensors',
        help="Path to a safetensors file that contains all model parts.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default='models/VACE-LTX-Video-0.9',
        help="Path to a safetensors file that contains all model parts.",
    )
    parser.add_argument(
        "--stg_mode",
        type=str,
        default="stg_a",
        help="Spatiotemporal guidance mode for the pipeline. Can be either stg_a or stg_r.",
    )
    parser.add_argument(
        "--stg_skip_layers",
        type=str,
        default="19",
        help="Attention layers to skip for spatiotemporal guidance. Comma separated list of integers.",
    )
    parser.add_argument(
        "--precision",
        choices=["bfloat16", "mixed_precision"],
        default="bfloat16",
        help="Sets the precision for the transformer and tokenizer. Default is bfloat16. If 'mixed_precision' is enabled, it moves to mixed-precision.",
    )
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offloading unnecessary computations to CPU.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with gr.Blocks() as demo:
        infer_gr = VACEInference(args, skip_load=False, gallery_share=True, gallery_share_limit=5)
        infer_gr.create_ui()
        infer_gr.set_callbacks()
        allowed_paths = [args.save_dir]
        demo.queue(status_update_rate=1).launch(server_name=args.server_name,
                                                server_port=args.server_port,
                                                root_path=args.root_path,
                                                allowed_paths=allowed_paths,
                                                show_error=True, debug=True)
