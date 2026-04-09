#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debugger.py
Main interface: run the model, collect per-node input/output tensors with
statistics, and return the structured result dict described in the design doc.

Result schema (per node):
{
  "<node_id>": {
    "op_type": str,
    "attrs":   dict,
    "inputs":  { "<tensor_name>": {shape, dtype, stats} },
    "outputs": { "<tensor_name>": {shape, dtype, stats} },
  }
}
"""

import onnx

from .core.runner import OnnxRunner
from .inspector.node_info import get_node_static_info
from .inspector.tensor_viewer import describe_tensor


class OnnxDebugger:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.runner = OnnxRunner(model_path)

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
    def run(self, npy_path: str) -> dict:
        """
        Run inference and return a per-node structured dict with actual tensor
        values (shapes + statistics) for every input and output.
        """
        all_tensors = self.runner.run_from_npy(npy_path)   # {name: np.ndarray}

        result = {}
        for idx, node in enumerate(self.model.graph.node):
            info = get_node_static_info(node, idx)
            node_id = info["node_id"]

            inputs_data = {}
            for name in info["input_names"]:
                if name:  # skip empty optional inputs
                    arr = all_tensors.get(name)
                    desc = describe_tensor(name, arr)
                    if desc.get("available"):
                        # Keep only the serialisable keys
                        inputs_data[name] = {
                            "shape": desc["shape"],
                            "dtype": desc["dtype"],
                            "stats": desc["stats"],
                        }
                    else:
                        inputs_data[name] = {"available": False}

            outputs_data = {}
            for name in info["output_names"]:
                if name:
                    arr = all_tensors.get(name)
                    desc = describe_tensor(name, arr)
                    if desc.get("available"):
                        outputs_data[name] = {
                            "shape": desc["shape"],
                            "dtype": desc["dtype"],
                            "stats": desc["stats"],
                        }
                    else:
                        outputs_data[name] = {"available": False}

            result[node_id] = {
                "op_type": info["op_type"],
                "attrs":   info["attrs"],
                "inputs":  inputs_data,
                "outputs": outputs_data,
            }

        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def inspect_node(self, node_id: str, npy_path: str):
        """Pretty-print a single node's I/O tensors."""
        all_results = self.run(npy_path)
        if node_id not in all_results:
            print(f"[ERR] Node '{node_id}' not found.  Available: {list(all_results.keys())[:5]} ...")
            return
        nd = all_results[node_id]
        print(f"\n=== Node: {node_id} ({nd['op_type']}) ===")
        for name, t in nd["inputs"].items():
            if t.get("available") is False:
                print(f"  INPUT  [{name}]: <not captured>")
            else:
                s = t["stats"]
                print(f"  INPUT  [{name}]: shape={t['shape']}, dtype={t['dtype']}, "
                      f"min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}")
        for name, t in nd["outputs"].items():
            if t.get("available") is False:
                print(f"  OUTPUT [{name}]: <not captured>")
            else:
                s = t["stats"]
                print(f"  OUTPUT [{name}]: shape={t['shape']}, dtype={t['dtype']}, "
                      f"min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}")
