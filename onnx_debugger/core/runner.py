#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/runner.py
Load a .npy input, run inference with all intermediate outputs exposed,
and return a flat dict of {tensor_name: np.ndarray}.
"""

import numpy as np
import onnxruntime as ort
import onnx

from .graph_patcher import patch_model_expose_all_intermediates


class OnnxRunner:
    def __init__(self, model_path: str):
        model = onnx.load(model_path)
        patched_model = patch_model_expose_all_intermediates(model)

        # Disable graph optimisation so nodes aren't fused/eliminated
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        # Load the modified model directly from memory — no temp file needed
        self.session = ort.InferenceSession(
            patched_model.SerializeToString(),
            sess_options=sess_options,
        )
        self.model = model

    def run_from_npy(self, npy_path: str) -> dict:
        """Load input.npy and return all tensors (inputs + every intermediate)."""
        input_data = np.load(npy_path, allow_pickle=True)

        # Support dict-in-npy (multiple inputs) or plain array (single input)
        if input_data.dtype == object:
            inputs = input_data.item()          # {name: array}
        else:
            input_name = self.session.get_inputs()[0].name
            inputs = {input_name: input_data}

        output_names = [o.name for o in self.session.get_outputs()]
        results = self.session.run(output_names, inputs)

        # Merge original inputs so every node's input tensors are reachable
        all_tensors = {**inputs, **dict(zip(output_names, results))}
        return all_tensors
