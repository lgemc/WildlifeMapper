"""
Wildlife Mapper Inference Module

This module provides inference functionality for wildlife detection and identification
using the Segment Anything Model (SAM) architecture.
"""

from .infer import (
    InferenceRunner,
    CocoEvaluator,
    evaluate,
    get_coco_api_from_dataset,
    convert_to_xywh,
    merge,
    create_common_coco_eval
)

__all__ = [
    'InferenceRunner',
    'CocoEvaluator',
    'evaluate',
    'get_coco_api_from_dataset',
    'convert_to_xywh',
    'merge',
    'create_common_coco_eval'
]