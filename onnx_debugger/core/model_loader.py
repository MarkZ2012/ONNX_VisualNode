#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/model_loader.py
Load an ONNX model and run shape inference.
"""

import onnx
import onnx.shape_inference


def load_model(model_path: str) -> onnx.ModelProto:
    """Load an ONNX model and attempt shape inference."""
    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[model_loader] shape inference warning: {e}")
    return model
