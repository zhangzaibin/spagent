# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Trimmed to only include the video preprocess config required by the
# firstframe pipeline (vace_pipeline.py --task=frameref --mode=firstframe).

from easydict import EasyDict


######################### R2V/MV2V - Extension #########################
# The 'mode' can be selected from options "firstframe", "lastframe",
# "firstlastframe"(needs image_2), "firstclip", "lastclip",
# "firstlastclip"(needs frames_2).
# "frames" refers to processing a video clip; 'image' refers to processing a single image.
#------------------------ FrameRefExp ------------------------#
video_framerefexp_anno = EasyDict()
video_framerefexp_anno.NAME = "FrameRefExpandAnnotator"
video_framerefexp_anno.INPUTS = {"image": None, "image_2": None, "mode": None, "expand_num": 80}
video_framerefexp_anno.OUTPUTS = {"frames": None, "masks": None}
