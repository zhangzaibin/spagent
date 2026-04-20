# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_t2v_1_3B import t2v_1_3B

WAN_CONFIGS = {
    'vace-1.3B': t2v_1_3B,
}

SIZE_CONFIGS = {
    '480*832': (480, 832),
    '832*480': (832, 480),
    '480p': (480, 832),
}

MAX_AREA_CONFIGS = {
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '480p': 480 * 832,
}

SUPPORTED_SIZES = {
    'vace-1.3B': ('480*832', '832*480', '480p'),
}
