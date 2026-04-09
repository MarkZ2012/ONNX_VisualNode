#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/graph_patcher.py
Inject all intermediate value_infos as graph outputs so ONNX Runtime
can return every intermediate tensor in one inference pass.
"""

import onnx


def patch_model_expose_all_intermediates(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Register every intermediate tensor (value_info) as a graph output.
    This is the core trick that lets us capture all node activations.
    """
    # Shape inference fills in value_info entries that may be missing
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    existing_outputs = {o.name for o in model.graph.output}

    for value_info in model.graph.value_info:
        if value_info.name not in existing_outputs:
            model.graph.output.append(value_info)

    return model
