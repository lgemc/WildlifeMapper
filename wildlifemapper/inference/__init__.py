"""
Inference module for WildlifeMapper
"""

from .evaluator import evaluate, get_coco_api_from_dataset

__all__ = ['evaluate', 'get_coco_api_from_dataset']