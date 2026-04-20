# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Trimmed to only include annotators used by the firstframe/frameref pipeline
# (see vace_pipeline.py --base wan --task frameref --mode firstframe).
# Uses lazy loading so adding more back later only requires extending the map.

import importlib


_CLASS_TO_MODULE = {
    # frameref (used by firstframe pipeline)
    "FrameRefExtractAnnotator": "frameref",
    "FrameRefExpandAnnotator": "frameref",
}


def __getattr__(name):
    module_name = _CLASS_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(f".{module_name}", __name__)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module '{module.__name__}' has no attribute '{name}'") from exc
    globals()[name] = value
    return value
