# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
#
# NOTE:
# Keep this module lightweight. The original eager imports pull in many optional
# dependencies (e.g. einops) even when a task only needs a small annotator set
# such as firstframe/frameref. We use lazy loading to import submodules only
# when the specific annotator class is requested.

import importlib


_CLASS_TO_MODULE = {
    # depth / flow / gray
    "DepthAnnotator": "depth",
    "DepthVideoAnnotator": "depth",
    "DepthV2VideoAnnotator": "depth",
    "FlowAnnotator": "flow",
    "FlowVisAnnotator": "flow",
    "GrayAnnotator": "gray",
    "GrayVideoAnnotator": "gray",
    # frameref
    "FrameRefExtractAnnotator": "frameref",
    "FrameRefExpandAnnotator": "frameref",
    # inpainting / layout / outpainting / composition
    "InpaintingAnnotator": "inpainting",
    "InpaintingVideoAnnotator": "inpainting",
    "LayoutBboxAnnotator": "layout",
    "LayoutMaskAnnotator": "layout",
    "LayoutTrackAnnotator": "layout",
    "OutpaintingAnnotator": "outpainting",
    "OutpaintingInnerAnnotator": "outpainting",
    "OutpaintingVideoAnnotator": "outpainting",
    "OutpaintingInnerVideoAnnotator": "outpainting",
    "CompositionAnnotator": "composition",
    "ReferenceAnythingAnnotator": "composition",
    "AnimateAnythingAnnotator": "composition",
    "SwapAnythingAnnotator": "composition",
    "ExpandAnythingAnnotator": "composition",
    "MoveAnythingAnnotator": "composition",
    # pose / face / subject
    "PoseBodyFaceAnnotator": "pose",
    "PoseBodyFaceVideoAnnotator": "pose",
    "PoseAnnotator": "pose",
    "PoseBodyVideoAnnotator": "pose",
    "PoseBodyAnnotator": "pose",
    "FaceAnnotator": "face",
    "SubjectAnnotator": "subject",
    # detection / segmentation related
    "GDINOAnnotator": "gdino",
    "GDINORAMAnnotator": "gdino",
    "SAMImageAnnotator": "sam",
    "SAM2ImageAnnotator": "sam2",
    "SAM2VideoAnnotator": "sam2",
    "SAM2SalientVideoAnnotator": "sam2",
    "SAM2GDINOVideoAnnotator": "sam2",
    "RAMAnnotator": "ram",
    "SalientAnnotator": "salient",
    "SalientVideoAnnotator": "salient",
    "MaskAugAnnotator": "maskaug",
    # common / prompt / misc
    "PlainImageAnnotator": "common",
    "PlainMaskAnnotator": "common",
    "PlainMaskAugAnnotator": "common",
    "PlainMaskVideoAnnotator": "common",
    "PlainVideoAnnotator": "common",
    "PlainMaskAugVideoAnnotator": "common",
    "PlainMaskAugInvertAnnotator": "common",
    "PlainMaskAugInvertVideoAnnotator": "common",
    "ExpandMaskVideoAnnotator": "common",
    "PromptExtendAnnotator": "prompt_extend",
    "MaskDrawAnnotator": "mask",
    "RegionCanvasAnnotator": "canvas",
    "ScribbleAnnotator": "scribble",
    "ScribbleVideoAnnotator": "scribble",
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