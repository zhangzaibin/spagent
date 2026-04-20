# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Trimmed to only the config actually used by the firstframe pipeline
# (vace_pipeline.py --base wan --task frameref --mode firstframe).
#
# The inference side defaults to --use_prompt_extend=plain which skips
# get_annotator() entirely, so no prompt config is needed here.

from .video_preproccess import video_framerefexp_anno


VACE_VIDEO_PREPROCCESS_CONFIGS = {
    'frameref': video_framerefexp_anno,
}


VACE_PREPROCCESS_CONFIGS = {**VACE_VIDEO_PREPROCCESS_CONFIGS}


VACE_CONFIGS = {
    "video": VACE_VIDEO_PREPROCCESS_CONFIGS,
}
