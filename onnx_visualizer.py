#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onnx_visualnode.py - ONNX Debugger Single File Version
A single-file version that combines all module functionalities for capturing and visualizing
intermediate tensor values of all nodes during ONNX model inference.

Usage:
    python onnx_visualnode.py resnet18.onnx
    python onnx_visualnode.py resnet18.onnx --output debug_report.html
    python onnx_visualnode.py resnet18.onnx input.npy
    python onnx_visualnode.py resnet18.onnx input.npy --output debug_report.html
    python onnx_visualnode.py resnet18.onnx input.npy --inspect Conv_0
"""

import sys
import os
import argparse
import json
import numpy as np
import onnx
import onnxruntime as ort
from datetime import datetime


# ============================================================================
# Module 1: graph_patcher - Graph Modification Tools
# ============================================================================

def patch_model_expose_all_intermediates(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Register all intermediate tensors (value_info) as graph outputs.
    This is the core technique for capturing all node activation values.
    """
    # Shape inference fills potentially missing value_info entries
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    existing_outputs = {o.name for o in model.graph.output}

    for value_info in model.graph.value_info:
        if value_info.name not in existing_outputs:
            model.graph.output.append(value_info)

    return model


# ============================================================================
# Module 2: model_loader - Model Loader
# ============================================================================

def load_model(model_path: str) -> onnx.ModelProto:
    """Load ONNX model and attempt shape inference."""
    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[model_loader] shape inference warning: {e}")
    return model


# ============================================================================
# Module 3: runner - Inference Runner
# ============================================================================

class OnnxRunner:
    def __init__(self, model_path: str):
        model = onnx.load(model_path)
        patched_model = patch_model_expose_all_intermediates(model)

        # Disable graph optimization to prevent node fusion
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        # Load modified model directly from memory - no temporary file needed
        self.session = ort.InferenceSession(
            patched_model.SerializeToString(),
            sess_options=sess_options,
        )
        self.model = model

    def run_from_npy(self, npy_path: str) -> dict:
        """Load input.npy and return all tensors (input + each intermediate layer)."""
        input_data = np.load(npy_path, allow_pickle=True)

        # Support dict-in-npy (multi-input) or plain array (single input)
        if input_data.dtype == object:
            inputs = input_data.item()          # {name: array}
        else:
            input_name = self.session.get_inputs()[0].name
            inputs = {input_name: input_data}

        output_names = [o.name for o in self.session.get_outputs()]
        results = self.session.run(output_names, inputs)

        # Merge original inputs so each node's input tensors are accessible
        all_tensors = {**inputs, **dict(zip(output_names, results))}
        return all_tensors


# ============================================================================
# Module 4: node_info - Node Information Extraction
# ============================================================================

def get_node_static_info(node: onnx.NodeProto, idx: int) -> dict:
    """Return {node_id, op_type, attrs, input_names, output_names}."""
    node_id = node.name if node.name else f"{node.op_type}_{idx}"

    attrs = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = round(attr.f, 7)
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode("utf-8", errors="replace")
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[attr.name] = [round(f, 7) for f in attr.floats]
        elif attr.type == onnx.AttributeProto.GRAPH:
            attrs[attr.name] = "<subgraph>"

    return {
        "node_id": node_id,
        "op_type": node.op_type,
        "attrs": attrs,
        "input_names": [n for n in node.input],
        "output_names": [n for n in node.output],
    }


# ============================================================================
# Module 5: tensor_viewer - Tensor Statistics
# ============================================================================

def tensor_stats(arr: np.ndarray) -> dict:
    """Return min/max/mean/std/abs_mean for numeric tensor."""
    if arr is None:
        return {}
    try:
        flat = arr.astype(np.float64).ravel()
        return {
            "min":      float(flat.min()),
            "max":      float(flat.max()),
            "mean":     float(flat.mean()),
            "std":      float(flat.std()),
            "abs_mean": float(np.abs(flat).mean()),
        }
    except Exception:
        return {}


def describe_tensor(name: str, arr: np.ndarray) -> dict:
    """
    Return detailed description dictionary for a single tensor:
    {shape, dtype, stats, has_nan, has_inf}
    """
    if arr is None:
        return {"name": name, "available": False}

    info = {
        "name":      name,
        "available": True,
        "shape":     list(arr.shape),
        "dtype":     str(arr.dtype),
        "stats":     tensor_stats(arr),
    }
    try:
        info["has_nan"] = bool(np.isnan(arr).any())
        info["has_inf"] = bool(np.isinf(arr).any())
    except Exception:
        info["has_nan"] = False
        info["has_inf"] = False
    return info


# ============================================================================
# Module 6: debugger - Main Debugger Interface
# ============================================================================

class OnnxDebugger:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.runner = OnnxRunner(model_path)

    # ------------------------------------------------------------------
    # Core Methods
    # ------------------------------------------------------------------
    def run(self, npy_path: str) -> dict:
        """
        Run inference and return structured dictionary for each node,
        containing actual tensor values (shape + statistics) for each input and output.
        """
        all_tensors = self.runner.run_from_npy(npy_path)   # {name: np.ndarray}

        result = {}
        for idx, node in enumerate(self.model.graph.node):
            info = get_node_static_info(node, idx)
            node_id = info["node_id"]

            inputs_data = {}
            for name in info["input_names"]:
                if name:  # Skip empty optional inputs
                    arr = all_tensors.get(name)
                    desc = describe_tensor(name, arr)
                    if desc.get("available"):
                        # Only keep serializable keys
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
    # Convenience Helper Methods
    # ------------------------------------------------------------------
    def inspect_node(self, node_id: str, npy_path: str):
        """Pretty print I/O tensors for a single node."""
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


# ============================================================================
# Module 7: html_builder - HTML Report Generation
# ============================================================================

def _safe_json(obj):
    """Serialize to JSON, gracefully converting non-serializable types."""
    return json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))


OP_CATEGORY = {
    "Conv": "conv", "ConvTranspose": "conv",
    "Gemm": "gemm", "MatMul": "gemm",
    "Relu": "act", "Sigmoid": "act", "Tanh": "act", "LeakyRelu": "act",
    "Elu": "act", "Selu": "act", "Softmax": "act", "Gelu": "act", "PRelu": "act",
    "MaxPool": "pool", "AveragePool": "pool", "GlobalAveragePool": "pool",
    "GlobalMaxPool": "pool",
    "BatchNormalization": "norm", "LayerNormalization": "norm",
    "InstanceNormalization": "norm",
    "Add": "eltwise", "Sub": "eltwise", "Mul": "eltwise", "Div": "eltwise",
    "Sum": "eltwise", "Max": "eltwise", "Min": "eltwise", "Pow": "eltwise",
    "Reshape": "shape", "Flatten": "shape", "Squeeze": "shape",
    "Unsqueeze": "shape", "Transpose": "shape", "Concat": "shape",
    "Slice": "shape", "Gather": "shape", "Expand": "shape", "Pad": "shape",
    "Resize": "upsample", "Upsample": "upsample",
    "LSTM": "rnn", "GRU": "rnn", "RNN": "rnn",
    "Dropout": "other", "Identity": "other", "Constant": "other",
    "Shape": "other", "Cast": "other", "Clip": "other",
    "ReduceMean": "reduce", "ReduceSum": "reduce", "ReduceMax": "reduce",
    "ReduceMin": "reduce", "ReduceL2": "reduce",
    "Attention": "attention", "MultiHeadAttention": "attention",
}

def _cat(op_type):
    return OP_CATEGORY.get(op_type, "other")


def _build_graph_data(debug_result: dict, model: onnx.ModelProto = None):
    """
    Build nodes_data and edges required for graph view from debug_result
    Returns: (nodes_data, edges)
    """
    nodes_data = []
    edges = []
    
    # Build mapping from tensor name to node id
    tensor_to_node = {}
    
    # Collect all input tensors (model inputs)
    all_input_tensors = set()
    all_output_tensors = set()
    all_intermediate_tensors = set()
    
    # Step 1: Collect all tensor information
    for idx, (node_id, nd) in enumerate(debug_result.items()):
        for inp_name in nd.get("inputs", {}).keys():
            if inp_name:
                all_input_tensors.add(inp_name)
        for out_name in nd.get("outputs", {}).keys():
            if out_name:
                all_output_tensors.add(out_name)
                all_intermediate_tensors.add(out_name)
    
    # Get real inputs from ONNX model (exclude weight parameters in initializer)
    model_inputs = set()
    if model is not None:
        # Get all initializer (weight parameter) names
        initializer_names = {init.name for init in model.graph.initializer}
        # Real model inputs = graph.input - initializer
        for input_info in model.graph.input:
            if input_info.name not in initializer_names:
                model_inputs.add(input_info.name)
    else:
        # If no model object, use old logic as fallback
        model_inputs = all_input_tensors - all_intermediate_tensors
    # Model outputs = all output tensors not used by other nodes
    model_outputs = set()
    for out_name in all_output_tensors:
        is_used = False
        for node_id, nd in debug_result.items():
            if out_name in nd.get("inputs", {}).keys():
                is_used = True
                break
        if not is_used:
            model_outputs.add(out_name)
    
    # Step 2: Create input nodes
    input_node_id = 0
    input_tensor_to_node = {}

    # Get input metadata from ONNX model
    input_metadata = {}
    if model is not None:
        initializer_names = {init.name for init in model.graph.initializer}
        for input_info in model.graph.input:
            if input_info.name not in initializer_names:
                # Extract shape and dtype information
                shape = []
                dtype_str = "unknown"
                if input_info.type.tensor_type:
                    tt = input_info.type.tensor_type
                    # Get shape
                    for dim in tt.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        elif dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(-1)
                    # Get dtype
                    if tt.elem_type:
                        from onnx import TensorProto
                        dtype_map = {
                            TensorProto.FLOAT: "float32",
                            TensorProto.DOUBLE: "float64",
                            TensorProto.INT32: "int32",
                            TensorProto.INT64: "int64",
                            TensorProto.UINT8: "uint8",
                            TensorProto.INT8: "int8",
                            TensorProto.UINT16: "uint16",
                            TensorProto.INT16: "int16",
                            TensorProto.BOOL: "bool",
                        }
                        dtype_str = dtype_map.get(tt.elem_type, f"type_{tt.elem_type}")
                input_metadata[input_info.name] = {"shape": shape, "dtype": dtype_str}

    for tensor_name in sorted(model_inputs):
        # Prefer getting shape from ONNX model metadata
        shape_str = "?"
        attrs = {}
        if tensor_name in input_metadata:
            meta = input_metadata[tensor_name]
            shape = meta.get("shape", [])
            if shape:
                shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
            attrs["dtype"] = meta.get("dtype", "unknown")
            attrs["shape"] = shape_str
        else:
            # fallback: get from debug_result
            for node_id, nd in debug_result.items():
                if tensor_name in nd.get("inputs", {}):
                    tensor_info = nd["inputs"][tensor_name]
                    shape = tensor_info.get("shape", [])
                    if shape:
                        shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
                    if tensor_info.get("dtype"):
                        attrs["dtype"] = tensor_info["dtype"]
                    attrs["shape"] = shape_str
                    break

        nodes_data.append({
            "id": input_node_id,
            "name": tensor_name[:40],
            "op": "Input",
            "category": "input",
            "inputs": [],
            "outputs": [tensor_name],
            "input_shapes": [],
            "output_shapes": [{"name": tensor_name[:40], "shape": shape_str}],
            "attrs": attrs,
        })
        input_tensor_to_node[tensor_name] = input_node_id
        input_node_id += 1
    
    # Step 3: Create all operator nodes
    for idx, (node_id, nd) in enumerate(debug_result.items()):
        nid = idx + input_node_id  # Offset by number of input nodes
        
        # Collect input and output shape information
        input_shapes = []
        for name, tensor_info in nd.get("inputs", {}).items():
            # Even if tensor is unavailable, try to get shape information
            shape = tensor_info.get("shape", [])
            if shape:
                shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
            else:
                shape_str = "?"
            input_shapes.append({"name": name[:40], "shape": shape_str})
        
        output_shapes = []
        for name, tensor_info in nd.get("outputs", {}).items():
            # Even if tensor is unavailable, try to get shape information
            shape = tensor_info.get("shape", [])
            if shape:
                shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
            else:
                shape_str = "?"
            output_shapes.append({"name": name[:40], "shape": shape_str})
        
        nodes_data.append({
            "id": nid,
            "name": node_id,
            "op": nd["op_type"],
            "category": _cat(nd["op_type"]),
            "inputs": list(nd.get("inputs", {}).keys()),
            "outputs": list(nd.get("outputs", {}).keys()),
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "attrs": nd.get("attrs", {}),
        })
        
        # Record output tensor to node mapping
        for out_name in nd.get("outputs", {}).keys():
            tensor_to_node[out_name] = nid
    
    # Step 4: Create output nodes
    output_node_start_id = len(nodes_data)
    output_tensor_to_node = {}

    # Get output metadata from ONNX model
    output_metadata = {}
    if model is not None:
        for output_info in model.graph.output:
            # Extract shape and dtype information
            shape = []
            dtype_str = "unknown"
            if output_info.type.tensor_type:
                tt = output_info.type.tensor_type
                # Get shape
                for dim in tt.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(-1)
                # Get dtype
                if tt.elem_type:
                    from onnx import TensorProto
                    dtype_map = {
                        TensorProto.FLOAT: "float32",
                        TensorProto.DOUBLE: "float64",
                        TensorProto.INT32: "int32",
                        TensorProto.INT64: "int64",
                        TensorProto.UINT8: "uint8",
                        TensorProto.INT8: "int8",
                        TensorProto.UINT16: "uint16",
                        TensorProto.INT16: "int16",
                        TensorProto.BOOL: "bool",
                    }
                    dtype_str = dtype_map.get(tt.elem_type, f"type_{tt.elem_type}")
            output_metadata[output_info.name] = {"shape": shape, "dtype": dtype_str}

    for tensor_name in sorted(model_outputs):
        # Prefer getting shape from ONNX model metadata
        shape_str = "?"
        attrs = {}
        if tensor_name in output_metadata:
            meta = output_metadata[tensor_name]
            shape = meta.get("shape", [])
            if shape:
                shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
            attrs["dtype"] = meta.get("dtype", "unknown")
            attrs["shape"] = shape_str
        else:
            # fallback: get from debug_result
            for node_id, nd in debug_result.items():
                if tensor_name in nd.get("outputs", {}):
                    tensor_info = nd["outputs"][tensor_name]
                    shape = tensor_info.get("shape", [])
                    if shape:
                        shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
                    if tensor_info.get("dtype"):
                        attrs["dtype"] = tensor_info["dtype"]
                    attrs["shape"] = shape_str
                    break

        output_node_id = output_node_start_id + len(output_tensor_to_node)
        nodes_data.append({
            "id": output_node_id,
            "name": tensor_name[:40],
            "op": "Output",
            "category": "output",
            "inputs": [tensor_name],
            "outputs": [],
            "input_shapes": [{"name": tensor_name[:40], "shape": shape_str}],
            "output_shapes": [],
            "attrs": attrs,
        })
        output_tensor_to_node[tensor_name] = output_node_id
    
    # Step 5: Build edges (based on tensor dependencies)
    seen_edges = set()
    
    # From input nodes to first operator node using them
    for tensor_name, src_id in input_tensor_to_node.items():
        for idx, (node_id, nd) in enumerate(debug_result.items()):
            if tensor_name in nd.get("inputs", {}).keys():
                dst_id = idx + input_node_id
                edge_key = (src_id, dst_id, tensor_name)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "src": src_id,
                        "dst": dst_id,
                        "tensor": tensor_name[:40]
                    })
    
    # Edges between operator nodes
    for idx, (node_id, nd) in enumerate(debug_result.items()):
        dst_id = idx + input_node_id
        for inp_name in nd.get("inputs", {}).keys():
            if inp_name in tensor_to_node:
                src_id = tensor_to_node[inp_name]
                edge_key = (src_id, dst_id, inp_name)
                if edge_key not in seen_edges and src_id != dst_id:
                    seen_edges.add(edge_key)
                    edges.append({
                        "src": src_id,
                        "dst": dst_id,
                        "tensor": inp_name[:40]
                    })
    
    # From operator nodes to output nodes
    for tensor_name, dst_id in output_tensor_to_node.items():
        if tensor_name in tensor_to_node:
            src_id = tensor_to_node[tensor_name]
            edge_key = (src_id, dst_id, tensor_name)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    "src": src_id,
                    "dst": dst_id,
                    "tensor": tensor_name[:40]
                })
    
    return nodes_data, edges


def build_html(
    debug_result: dict,
    model_path: str,
    npy_path: str = None,
    output_path: str = None,
):
    """
    debug_result  – Dictionary returned by OnnxDebugger.run()
    model_path    – .onnx path (for display)
    npy_path      – Input .npy path (for display, optional)
    output_path   – .html file write location
    """
    # Load model to get correct input information
    model = onnx.load(model_path)

    # Add id and category to each node dictionary for JS use
    nodes_list = []
    for node_id, nd in debug_result.items():
        entry = {
            "node_id":  node_id,
            "op_type":  nd["op_type"],
            "category": _cat(nd["op_type"]),
            "attrs":    nd.get("attrs", {}),
            "inputs":   nd.get("inputs", {}),
            "outputs":  nd.get("outputs", {}),
        }
        nodes_list.append(entry)

    # Prepare graph view data (nodes_data and edges)
    nodes_data, edges = _build_graph_data(debug_result, model)

    data_json = _safe_json({
        "model":    os.path.basename(model_path),
        "npy":      os.path.basename(npy_path) if npy_path else "No input file",
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nodes":    nodes_list,
        "graph_nodes": nodes_data,
        "graph_edges": edges,
    })

    html = _render_html(data_json, os.path.basename(model_path))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Report written -> {output_path}")


# ─── HTML模板 ─────────────────────────────────────────────────────────────

def _render_html(data_json: str, model_name: str) -> str:  # noqa: C901
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ONNX Debugger - {model_name}</title>
<style>
:root{{
  --bg:#0d1117;--panel:#161b22;--panel2:#1f2937;--border:#2d3748;
  --accent:#3b82f6;--accent2:#8b5cf6;--text:#e2e8f0;--text2:#94a3b8;
  --success:#22c55e;--warn:#f59e0b;--danger:#ef4444;--info:#06b6d4;
  --c-conv:#3b82f6;--c-gemm:#8b5cf6;--c-act:#22c55e;--c-pool:#06b6d4;
  --c-norm:#f59e0b;--c-eltwise:#ec4899;--c-shape:#a78bfa;
  --c-upsample:#10b981;--c-rnn:#f97316;--c-reduce:#84cc16;
  --c-attention:#e11d48;--c-other:#6b7280;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
     height:100vh;display:flex;flex-direction:column;overflow:hidden}}
a{{color:var(--accent);text-decoration:none}}

/* ── header ── */
header{{background:var(--panel);border-bottom:1px solid var(--border);
       padding:10px 20px;display:flex;align-items:center;gap:14px;flex-shrink:0}}
.logo{{font-size:19px;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;white-space:nowrap}}
.hinfo{{font-size:12px;color:var(--text2);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.hchip{{background:var(--panel2);border:1px solid var(--border);border-radius:6px;
       padding:3px 10px;font-size:12px;white-space:nowrap}}
.hchip span{{color:var(--accent);font-weight:700}}
.tabs{{display:flex;gap:3px}}
.tab{{padding:5px 14px;border-radius:6px;cursor:pointer;font-size:13px;color:var(--text2);
     border:1px solid transparent;transition:all .15s}}
.tab.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.tab:hover:not(.active){{background:var(--panel2);color:var(--text)}}

/* ── layout ── */
.main{{display:flex;flex:1;overflow:hidden}}

/* ── node list (left) ── */
#list-panel{{width:300px;background:var(--panel);border-right:1px solid var(--border);
            display:flex;flex-direction:column;flex-shrink:0}}
#search-wrap{{padding:10px;border-bottom:1px solid var(--border)}}
#node-search{{width:100%;background:var(--panel2);border:1px solid var(--border);
             border-radius:8px;color:var(--text);padding:6px 12px;font-size:13px;outline:none}}
#node-search:focus{{border-color:var(--accent)}}
#node-list{{flex:1;overflow-y:auto;padding:6px}}
#node-list::-webkit-scrollbar{{width:4px}}
#node-list::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
.node-item{{padding:7px 10px;border-radius:8px;cursor:pointer;margin-bottom:3px;
           display:flex;align-items:center;gap:8px;transition:background .12s}}
.node-item:hover{{background:var(--panel2)}}
.node-item.active{{background:var(--panel2);border:1px solid var(--accent)}}
.op-dot{{width:9px;height:9px;border-radius:3px;flex-shrink:0}}
.node-item-info{{min-width:0}}
.ni-op{{font-size:12px;font-weight:700;color:var(--text)}}
.ni-id{{font-size:10px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}

/* ── detail panel (right) ── */
#detail-panel{{flex:1;overflow-y:auto;padding:16px;background:var(--bg)}}
#detail-panel::-webkit-scrollbar{{width:6px}}
#detail-panel::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}
.no-sel{{color:var(--text2);text-align:center;padding:60px 20px;font-size:14px}}

/* card */
.card{{background:var(--panel);border:1px solid var(--border);border-radius:12px;
      padding:16px;margin-bottom:14px}}
.card-title{{font-size:12px;font-weight:700;color:var(--text2);text-transform:uppercase;
            letter-spacing:1px;margin-bottom:12px;display:flex;align-items:center;gap:6px}}
.card-title .icon{{font-size:14px}}

.op-badge{{display:inline-block;border-radius:8px;padding:4px 14px;font-size:15px;
          font-weight:700;margin-bottom:10px}}
.info-row{{display:flex;gap:8px;margin-bottom:6px;font-size:12px}}
.ik{{color:var(--text2);width:90px;flex-shrink:0}}
.iv{{color:var(--text);word-break:break-all;flex:1}}

/* tensor grid */
.tensors-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
@media(max-width:900px){{.tensors-grid{{grid-template-columns:1fr}}}}
.tensor-card{{background:var(--panel2);border-radius:8px;padding:10px 12px;border:1px solid var(--border)}}
.tensor-name{{font-size:11px;font-weight:700;color:var(--accent);margin-bottom:6px;
             word-break:break-all}}
.tensor-meta{{font-size:11px;color:var(--text2);margin-bottom:6px}}
.stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:3px 10px}}
.stat-row{{display:flex;justify-content:space-between;font-size:11px}}
.sk{{color:var(--text2)}}
.sv{{color:var(--text);font-weight:600}}
.sv.danger{{color:var(--danger)}}
.sv.warn{{color:var(--warn)}}
.badge-nan{{background:var(--danger);color:#fff;font-size:9px;border-radius:4px;
          padding:1px 5px;margin-left:4px}}
.badge-inf{{background:var(--warn);color:#000;font-size:9px;border-radius:4px;
          padding:1px 5px;margin-left:4px}}

/* attr table */
.attr-table{{width:100%;border-collapse:collapse;font-size:12px}}
.attr-table td{{padding:4px 8px;border-bottom:1px solid rgba(45,55,72,.5)}}
.attr-table td:first-child{{color:var(--warn);width:120px;vertical-align:top}}

/* ── stats tab ── */
#stats-panel{{display:none;flex:1;overflow-y:auto;padding:20px;background:var(--bg)}}
#stats-panel.active{{display:block}}
.scard-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px}}
.scard{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px}}
.scard-label{{font-size:12px;color:var(--text2);margin-bottom:8px}}
.scard-val{{font-size:26px;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.scard-sub{{font-size:11px;color:var(--text2);margin-top:4px}}
.op-table{{width:100%;border-collapse:collapse;font-size:13px}}
.op-table th{{text-align:left;padding:8px 12px;font-size:11px;color:var(--text2);
             text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border)}}
.op-table td{{padding:7px 12px;border-bottom:1px solid rgba(45,55,72,.4)}}
.op-table tr:hover td{{background:var(--panel)}}
.bar-bg{{background:var(--panel2);border-radius:4px;height:5px;min-width:80px}}
.bar-fill{{height:5px;border-radius:4px}}

/* ── graph tab ── */
#graph-panel{{display:none;flex:1;overflow:hidden;position:relative;background:var(--bg)}}
#graph-panel.active{{display:flex}}
#canvas-wrap{{flex:1;position:relative;overflow:hidden;cursor:grab}}
#canvas-wrap.grabbing{{cursor:grabbing}}
svg#graph-svg{{width:100%;height:100%}}
.node-group{{cursor:pointer;transition:filter .15s}}
.node-group:hover{{filter:brightness(1.3)}}
.node-rect{{rx:8;ry:8;stroke-width:1.5}}
.node-op{{font-size:12px;font-weight:700;fill:#fff;text-anchor:middle;dominant-baseline:central;pointer-events:none}}
.node-name{{font-size:9px;fill:rgba(255,255,255,0.6);text-anchor:middle;pointer-events:none}}
.edge-path{{fill:none;stroke:#3a3f5c;stroke-width:1.2;marker-end:url(#arrow);opacity:0.6}}
.edge-path.highlighted{{stroke:var(--accent);opacity:1;stroke-width:2}}
.node-rect.selected{{stroke:#fff;stroke-width:3;filter:drop-shadow(0 0 8px var(--accent))}}

.graph-controls{{position:absolute;bottom:16px;left:16px;display:flex;flex-direction:column;gap:6px}}
.ctrl-btn{{background:var(--panel);border:1px solid var(--border);border-radius:6px;color:var(--text);
          width:32px;height:32px;cursor:pointer;font-size:16px;display:flex;align-items:center;
          justify-content:center;transition:background .2s}}
.ctrl-btn:hover{{background:var(--panel2)}}
.graph-search{{position:absolute;top:12px;left:12px}}
#graph-search{{background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);
              padding:6px 12px;font-size:13px;width:220px;outline:none}}
#graph-search:focus{{border-color:var(--accent)}}

.graph-legend{{position:absolute;top:12px;right:12px;background:rgba(22,27,34,0.92);
              border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-size:11px;
              display:flex;flex-direction:column;gap:4px}}
.legend-item{{display:flex;align-items:center;gap:6px}}
.legend-dot{{width:10px;height:10px;border-radius:3px;flex-shrink:0}}

#graph-detail-panel{{width:320px;background:var(--panel);border-left:1px solid var(--border);
                    display:none;flex-direction:column;flex-shrink:0;overflow:hidden;position:absolute;right:0;top:0;bottom:0;z-index:10}}
#graph-detail-panel.visible{{display:flex}}
.panel-header{{padding:12px 16px;border-bottom:1px solid var(--border);font-size:13px;
              font-weight:600;color:var(--text2);display:flex;align-items:center;justify-content:space-between}}
.panel-content{{flex:1;overflow-y:auto;padding:12px}}
.panel-content::-webkit-scrollbar{{width:4px}}
.panel-content::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
.section-title{{font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;
              letter-spacing:1px;margin:12px 0 6px}}
.tensor-item{{background:var(--panel2);border-radius:6px;padding:6px 10px;margin-bottom:4px;font-size:11px}}
.attr-item{{display:flex;gap:6px;font-size:11px;margin-bottom:4px}}
.attr-key{{color:var(--warn);min-width:80px;flex-shrink:0}}
.attr-val{{color:var(--text);word-break:break-all}}
</style>
</head>
<body>

<header>
  <div class="logo">🐛 ONNX Debugger</div>
  <div class="hinfo" id="h-model"></div>
  <div class="hchip">Nodes <span id="h-nodes">—</span></div>
  <div class="hchip">Time <span id="h-time">—</span></div>
  <div class="tabs">
    <div class="tab active" id="tab-debug" onclick="switchTab('debug')">Debug View</div>
    <div class="tab" id="tab-graph" onclick="switchTab('graph')">Graph View</div>
    <div class="tab" id="tab-stats" onclick="switchTab('stats')">Statistics</div>
  </div>
</header>

<div class="main">
  <!-- left: node list (for debug view) -->
  <div id="list-panel">
    <div id="search-wrap">
      <input id="node-search" type="text" placeholder="🔍 Filter nodes…">
    </div>
    <div id="node-list"></div>
  </div>

  <!-- right: detail + stats + graph -->
  <div style="flex:1;display:flex;flex-direction:column;overflow:hidden">
    <div id="detail-panel">
      <div class="no-sel">← Select a node to inspect its tensors</div>
    </div>
    <div id="stats-panel"></div>
    <div id="graph-panel">
      <div id="canvas-wrap">
        <svg id="graph-svg">
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#3a3f5c"/>
            </marker>
            <marker id="arrow-hl" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="var(--accent)"/>
            </marker>
          </defs>
          <g id="graph-root"></g>
        </svg>
      </div>
      <div class="graph-search">
        <input id="graph-search" type="text" placeholder="🔍 Search nodes...">
      </div>
      <div class="graph-controls">
        <button class="ctrl-btn" title="Zoom In" onclick="zoomBy(1.25)">+</button>
        <button class="ctrl-btn" title="Zoom Out" onclick="zoomBy(0.8)">−</button>
        <button class="ctrl-btn" title="Fit View" onclick="fitView()" style="font-size:13px">⛶</button>
        <button class="ctrl-btn" title="Toggle Panel" onclick="toggleGraphPanel()" style="font-size:13px">☰</button>
      </div>
      <div class="graph-legend" id="legend-panel"></div>
      <div id="graph-detail-panel" class="hidden">
        <div class="panel-header">
          <span>Node Details</span>
          <span style="font-size:10px;cursor:pointer;color:var(--text2)" onclick="toggleGraphPanel()">✕</span>
        </div>
        <div class="panel-content" id="graph-node-detail">
          <div class="no-sel">Click a node to view details</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const RAW = {data_json};
const nodes = RAW.nodes;
const graphNodes = RAW.graph_nodes || [];
const graphEdges = RAW.graph_edges || [];

// ── colour map ──────────────────────────────────────────────────────────────
const CAT_COLOR = {{
  conv:"#3b82f6",gemm:"#8b5cf6",act:"#22c55e",pool:"#06b6d4",
  norm:"#f59e0b",eltwise:"#ec4899",shape:"#a78bfa",upsample:"#10b981",
  rnn:"#f97316",reduce:"#84cc16",attention:"#e11d48",other:"#6b7280"
}};
const CAT_LABEL = {{
  conv:"Conv",gemm:"FC/MatMul",act:"Activation",pool:"Pooling",
  norm:"Normalization",eltwise:"Element-wise",shape:"Shape Ops",
  upsample:"Upsample",rnn:"RNN",reduce:"Reduce",attention:"Attention",
  input:"Input",output:"Output",other:"Other"
}};
function catColor(c){{ return CAT_COLOR[c]||"#6b7280"; }}

// ── header info ─────────────────────────────────────────────────────────────
document.getElementById("h-model").textContent = RAW.model + "  ·  " + RAW.npy;
document.getElementById("h-nodes").textContent = nodes.length;
document.getElementById("h-time").textContent  = RAW.time;

// ═══════════════════════════════════════════════════════════════════════════
// DEBUG VIEW
// ═══════════════════════════════════════════════════════════════════════════

// ── build node list ─────────────────────────────────────────────────────────
function buildList(filter){{
  const container = document.getElementById("node-list");
  container.innerHTML = "";
  const q = (filter||"").toLowerCase();
  nodes.forEach((nd,i) => {{
    if(q && !nd.op_type.toLowerCase().includes(q) && !nd.node_id.toLowerCase().includes(q)) return;
    const div = document.createElement("div");
    div.className = "node-item" + (i===activeIdx?" active":"");
    div.dataset.idx = i;
    const color = catColor(nd.category);
    div.innerHTML = `
      <div class="op-dot" style="background:${{color}}"></div>
      <div class="node-item-info" style="min-width:0">
        <div class="ni-op" style="color:${{color}}">${{nd.op_type}}</div>
        <div class="ni-id" title="${{nd.node_id}}">${{nd.node_id}}</div>
      </div>`;
    div.addEventListener("click", () => selectNode(i));
    container.appendChild(div);
  }});
}}

document.getElementById("node-search").addEventListener("input", function(){{
  buildList(this.value);
}});

// ── select node ─────────────────────────────────────────────────────────────
let activeIdx = -1;
function selectNode(idx){{
  activeIdx = idx;
  buildList(document.getElementById("node-search").value);
  renderDetail(nodes[idx]);
}}

// ── format helpers ──────────────────────────────────────────────────────────
function fmt4(v){{
  if(v===undefined||v===null) return "—";
  return (typeof v==="number") ? v.toPrecision(6) : String(v);
}}

function tensorCard(name, t){{
  if(!t || t.available===false) return `
    <div class="tensor-card">
      <div class="tensor-name">${{name}}</div>
      <div class="tensor-meta" style="color:var(--danger)">Not captured</div>
    </div>`;

  const nanBadge = t.has_nan ? `<span class="badge-nan">NaN</span>`:"";
  const infBadge = t.has_inf ? `<span class="badge-inf">Inf</span>`:"";
  const s = t.stats||{{}};
  return `
    <div class="tensor-card">
      <div class="tensor-name">${{name}}${{nanBadge}}${{infBadge}}</div>
      <div class="tensor-meta">shape: [${{(t.shape||[]).join(", ")}}] &nbsp;·&nbsp; dtype: ${{t.dtype||"?"}}</div>
      <div class="stat-grid">
        <div class="stat-row"><span class="sk">min</span><span class="sv">${{fmt4(s.min)}}</span></div>
        <div class="stat-row"><span class="sk">max</span><span class="sv">${{fmt4(s.max)}}</span></div>
        <div class="stat-row"><span class="sk">mean</span><span class="sv">${{fmt4(s.mean)}}</span></div>
        <div class="stat-row"><span class="sk">std</span><span class="sv">${{fmt4(s.std)}}</span></div>
        <div class="stat-row"><span class="sk">abs_mean</span><span class="sv">${{fmt4(s.abs_mean)}}</span></div>
      </div>
    </div>`;
}}

function renderDetail(nd){{
  const color = catColor(nd.category);
  let html = `
    <div class="card">
      <div class="op-badge" style="background:${{color}}22;border:1px solid ${{color}};color:${{color}}">${{nd.op_type}}</div>
      <div class="info-row"><div class="ik">Node ID</div><div class="iv">${{nd.node_id}}</div></div>
      <div class="info-row"><div class="ik">Category</div><div class="iv">${{CAT_LABEL[nd.category]||nd.category}}</div></div>
    </div>`;

  // Attributes
  const attrEntries = Object.entries(nd.attrs||{{}});
  if(attrEntries.length){{
    html += `<div class="card"><div class="card-title"><span class="icon">⚙️</span>Attributes</div>
      <table class="attr-table">`;
    attrEntries.forEach(([k,v])=>{{
      html += `<tr><td>${{k}}</td><td>${{JSON.stringify(v)}}</td></tr>`;
    }});
    html += `</table></div>`;
  }}

  // Inputs
  const inEntries = Object.entries(nd.inputs||{{}});
  if(inEntries.length){{
    html += `<div class="card"><div class="card-title"><span class="icon">📥</span>Input Tensors (` + inEntries.length + `)</div>
      <div class="tensors-grid">`;
    inEntries.forEach(([name,t])=>{{ html += tensorCard(name,t); }});
    html += `</div></div>`;
  }}

  // Outputs
  const outEntries = Object.entries(nd.outputs||{{}});
  if(outEntries.length){{
    html += `<div class="card"><div class="card-title"><span class="icon">📤</span>Output Tensors (` + outEntries.length + `)</div>
      <div class="tensors-grid">`;
    outEntries.forEach(([name,t])=>{{ html += tensorCard(name,t); }});
    html += `</div></div>`;
  }}

  document.getElementById("detail-panel").innerHTML = html;
}}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS TAB
// ═══════════════════════════════════════════════════════════════════════════

function buildStats(){{
  const counter = {{}};
  nodes.forEach(nd=>{{ counter[nd.op_type] = (counter[nd.op_type]||0)+1; }});
  const sorted = Object.entries(counter).sort((a,b)=>b[1]-a[1]);
  const maxCnt = sorted.length ? sorted[0][1] : 1;

  let html = `<div class="scard-grid">
    <div class="scard"><div class="scard-label">Total Nodes</div>
      <div class="scard-val">${{nodes.length}}</div>
      <div class="scard-sub">operator nodes</div></div>
    <div class="scard"><div class="scard-label">Unique Op Types</div>
      <div class="scard-val">${{sorted.length}}</div>
      <div class="scard-sub">distinct operators</div></div>
    <div class="scard"><div class="scard-label">Analyzed At</div>
      <div class="scard-val" style="font-size:14px;padding-top:6px">${{RAW.time}}</div>
      <div class="scard-sub">${{RAW.model}}</div></div>
  </div>`;

  html += `<div class="card"><div class="card-title">Operator Breakdown</div>
  <table class="op-table">
    <tr><th>Operator</th><th>Category</th><th>Count</th><th>Bar</th></tr>`;
  sorted.forEach(([op,cnt])=>{{
    const cat = nodes.find(n=>n.op_type===op)?.category||"other";
    const color = catColor(cat);
    const pct = (cnt/nodes.length*100).toFixed(1);
    const barW = Math.round(cnt/maxCnt*100);
    html += `<tr>
      <td><span style="display:inline-block;width:9px;height:9px;border-radius:3px;
                        background:${{color}};margin-right:7px;vertical-align:middle"></span>${{op}}</td>
      <td style="color:var(--text2)">${{CAT_LABEL[cat]||cat}}</td>
      <td style="font-weight:700">${{cnt}} <span style="color:var(--text2);font-size:11px">(${{pct}}%)</span></td>
      <td><div class="bar-bg"><div class="bar-fill" style="width:${{barW}}%;background:${{color}}"></div></div></td>
    </tr>`;
  }});
  html += `</table></div>`;

  document.getElementById("stats-panel").innerHTML = html;
}}

// ═══════════════════════════════════════════════════════════════════════════
// GRAPH VIEW
// ═══════════════════════════════════════════════════════════════════════════

const NODE_W = 130, NODE_H = 46, LEVEL_GAP = 80, COL_GAP = 150;

function computeLayout(nodes, edges) {{
  const n = nodes.length;
  if(n === 0) return;

  const idToIdx = {{}};
  nodes.forEach((nd, i) => idToIdx[nd.id] = i);

  const inDeg = new Array(n).fill(0);
  const adj = Array.from({{length: n}}, () => []);

  edges.forEach(e => {{
    const si = idToIdx[e.src], di = idToIdx[e.dst];
    if(si === undefined || di === undefined || si === di) return;
    adj[si].push(di);
    inDeg[di]++;
  }});

  const level = new Array(n).fill(0);
  const queue = [];
  for(let i=0;i<n;i++) if(inDeg[i]===0) queue.push(i);

  const topo = [];
  const visited = new Array(n).fill(false);
  let qi = 0;
  while(qi < queue.length) {{
    const u = queue[qi++];
    topo.push(u);
    visited[u] = true;
    adj[u].forEach(v => {{
      level[v] = Math.max(level[v], level[u]+1);
      inDeg[v]--;
      if(inDeg[v] === 0) queue.push(v);
    }});
  }}
  for(let i=0;i<n;i++) if(!visited[i]) {{ topo.push(i); }}

  const maxLevel = Math.max(...level, 0);
  const levelGroups = Array.from({{length: maxLevel+1}}, () => []);
  topo.forEach(i => levelGroups[level[i]].push(i));

  nodes.forEach((nd, i) => {{
    const lv = level[i];
    const grp = levelGroups[lv];
    const posInGrp = grp.indexOf(i);
    const totalInGrp = grp.length;
    nd._w = NODE_W; nd._h = NODE_H;
    nd._level = lv;
    nd.y = lv * (NODE_H + LEVEL_GAP) + 40;
    nd.x = (posInGrp - (totalInGrp-1)/2) * (NODE_W + COL_GAP);
  }});

  nodes._levelGroups = levelGroups;
  nodes._idToIdx = idToIdx;
  nodes._level = level;
}}

computeLayout(graphNodes, graphEdges);

// center offset
let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
graphNodes.forEach(nd => {{
  if(nd.x !== undefined) {{
    minX = Math.min(minX, nd.x); minY = Math.min(minY, nd.y);
    maxX = Math.max(maxX, nd.x + NODE_W); maxY = Math.max(maxY, nd.y + NODE_H);
  }}
}});
const graphW = maxX - minX + NODE_W*2;
const graphH = maxY - minY + NODE_H*2;
const offsetX = -minX + NODE_W;
const offsetY = -minY + NODE_H;
graphNodes.forEach(nd => {{ if(nd.x !== undefined) {{ nd.x += offsetX; nd.y += offsetY; }} }});

// ── SVG rendering ──
const svg = document.getElementById('graph-svg');
const root = document.getElementById('graph-root');
let transform = {{ x: 0, y: 0, scale: 1 }};

function applyTransform() {{
  root.setAttribute('transform', `translate(${{transform.x}},${{transform.y}}) scale(${{transform.scale}})`);
}}

function buildGraph() {{
  root.innerHTML = '';

  const nodeById = {{}};
  graphNodes.forEach(nd => nodeById[nd.id] = nd);

  // Build edge lists
  const outEdges = {{}};
  const inEdges  = {{}};
  graphEdges.forEach((e, ei) => {{
    if(!outEdges[e.src]) outEdges[e.src] = [];
    if(!inEdges[e.dst])  inEdges[e.dst]  = [];
    outEdges[e.src].push(ei);
    inEdges[e.dst].push(ei);
  }});

  // Sort edges
  Object.keys(outEdges).forEach(srcId => {{
    outEdges[+srcId].sort((a, b) => {{
      const da = nodeById[graphEdges[a].dst], db = nodeById[graphEdges[b].dst];
      if(!da || !db) return 0;
      return (da.x + NODE_W/2) - (db.x + NODE_W/2);
    }});
  }});
  Object.keys(inEdges).forEach(dstId => {{
    inEdges[+dstId].sort((a, b) => {{
      const sa = nodeById[graphEdges[a].src], sb = nodeById[graphEdges[b].src];
      if(!sa || !sb) return 0;
      return (sa.x + NODE_W/2) - (sb.x + NODE_W/2);
    }});
  }});

  function anchorX(nodeId, edgeIdx, side) {{
    const nd = nodeById[nodeId];
    if(!nd) return 0;
    const list = side === 'out' ? (outEdges[nodeId] || []) : (inEdges[nodeId] || []);
    const total = list.length;
    if(total <= 1) return nd.x + NODE_W / 2;
    const pos = list.indexOf(edgeIdx);
    const margin = NODE_W * 0.10;
    const span   = NODE_W - margin * 2;
    return nd.x + margin + (pos / (total - 1)) * span;
  }}

  // Draw edges
  const edgeG = document.createElementNS('http://www.w3.org/2000/svg','g');
  edgeG.id = 'edges-g';
  graphEdges.forEach((e, ei) => {{
    const src = nodeById[e.src], dst = nodeById[e.dst];
    if(!src || !dst || src.x === undefined || dst.x === undefined) return;

    const sx = anchorX(e.src, ei, 'out');
    const sy = src.y + NODE_H;
    const dx = anchorX(e.dst, ei, 'in');
    const dy = dst.y;

    const levelSpan = (dst._level || 0) - (src._level || 0);
    let pathD;

    if(levelSpan > 1) {{
      // Long edge: bypass routing
      const goLeft = sx <= (src.x + NODE_W/2);
      const bounds = {{minX: Infinity, maxX: -Infinity}};
      for(let lv = src._level; lv <= dst._level; lv++) {{
        graphNodes.forEach(nd => {{
          if(nd._level === lv) {{
            bounds.minX = Math.min(bounds.minX, nd.x);
            bounds.maxX = Math.max(bounds.maxX, nd.x + NODE_W);
          }}
        }});
      }}
      if(!isFinite(bounds.minX)) bounds.minX = 0;
      if(!isFinite(bounds.maxX)) bounds.maxX = 0;

      const BYPASS_MARGIN = 18;
      const STUB = 20;
      if(goLeft) {{
        const lx = bounds.minX - BYPASS_MARGIN - ei * 14;
        pathD = `M${{sx}},${{sy}} L${{sx}},${{sy+STUB}} L${{lx}},${{sy+STUB}} L${{lx}},${{dy-STUB}} L${{dx}},${{dy-STUB}} L${{dx}},${{dy}}`;
      }} else {{
        const rx = bounds.maxX + BYPASS_MARGIN + ei * 14;
        pathD = `M${{sx}},${{sy}} L${{sx}},${{sy+STUB}} L${{rx}},${{sy+STUB}} L${{rx}},${{dy-STUB}} L${{dx}},${{dy-STUB}} L${{dx}},${{dy}}`;
      }}
    }} else {{
      // Short edge: simple bezier
      const cp = Math.max(Math.abs(dy - sy) * 0.5, 30);
      pathD = `M${{sx}},${{sy}} C${{sx}},${{sy + cp}} ${{dx}},${{dy - cp}} ${{dx}},${{dy}}`;
    }}

    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d', pathD);
    path.setAttribute('class','edge-path');
    path.setAttribute('data-src', e.src);
    path.setAttribute('data-dst', e.dst);
    edgeG.appendChild(path);
  }});
  root.appendChild(edgeG);

  // Draw nodes
  const nodeG = document.createElementNS('http://www.w3.org/2000/svg','g');
  nodeG.id = 'nodes-g';
  graphNodes.forEach((nd, i) => {{
    if(nd.x === undefined) return;
    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    g.setAttribute('class','node-group');
    g.setAttribute('data-id', nd.id);
    g.setAttribute('transform', `translate(${{nd.x}},${{nd.y}})`);

    const color = catColor(nd.category);

    const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('class','node-rect');
    rect.setAttribute('width', NODE_W);
    rect.setAttribute('height', NODE_H);
    rect.setAttribute('fill', color+'22');
    rect.setAttribute('stroke', color);
    rect.setAttribute('rx', 8);
    g.appendChild(rect);

    const opText = document.createElementNS('http://www.w3.org/2000/svg','text');
    opText.setAttribute('class','node-op');
    opText.setAttribute('x', NODE_W/2);
    opText.setAttribute('y', 18);
    opText.textContent = nd.op.length > 15 ? nd.op.slice(0,14)+'…' : nd.op;
    opText.setAttribute('fill', color);
    g.appendChild(opText);

    const nameText = document.createElementNS('http://www.w3.org/2000/svg','text');
    nameText.setAttribute('class','node-name');
    nameText.setAttribute('x', NODE_W/2);
    nameText.setAttribute('y', 36);
    const dispName = nd.name.length > 18 ? nd.name.slice(0,17)+'…' : nd.name;
    nameText.textContent = dispName;
    g.appendChild(nameText);

    g.addEventListener('click', () => selectGraphNode(nd.id));
    nodeG.appendChild(g);
  }});
  root.appendChild(nodeG);
}}

// ── Legend ──
function buildLegend() {{
  const usedCats = [...new Set(graphNodes.map(n => n.category))];
  const leg = document.getElementById('legend-panel');
  leg.innerHTML = usedCats.map(c =>
    `<div class="legend-item"><div class="legend-dot" style="background:${{catColor(c)}}"></div><span>${{CAT_LABEL[c]||c}}</span></div>`
  ).join('');
}}

// ── Node selection ──
let selectedGraphNode = null;
function selectGraphNode(nodeId) {{
  // deselect previous
  if(selectedGraphNode !== null) {{
    const prev = root.querySelector(`[data-id="${{selectedGraphNode}}"] .node-rect`);
    if(prev) prev.classList.remove('selected');
  }}
  // highlight edges
  root.querySelectorAll('.edge-path').forEach(p => {{
    p.classList.remove('highlighted');
    p.setAttribute('marker-end','url(#arrow)');
  }});

  selectedGraphNode = nodeId;
  const el = root.querySelector(`[data-id="${{nodeId}}"] .node-rect`);
  if(el) el.classList.add('selected');

  // highlight connected edges
  root.querySelectorAll(`.edge-path[data-src="${{nodeId}}"], .edge-path[data-dst="${{nodeId}}"]`).forEach(p => {{
    p.classList.add('highlighted');
    p.setAttribute('marker-end','url(#arrow-hl)');
  }});

  const nd = graphNodes.find(n => n.id === nodeId);
  if(nd) renderGraphNodeDetail(nd);
  
  // Show the detail panel
  graphPanelVisible = true;
  document.getElementById('graph-detail-panel').classList.add('visible');
}}

function renderGraphNodeDetail(nd) {{
  const color = catColor(nd.category);

  // Special handling for Input and Output node category labels
  let categoryLabel = CAT_LABEL[nd.category] || nd.category;
  if(nd.op === 'Input') categoryLabel = 'Input';
  if(nd.op === 'Output') categoryLabel = 'Output';

  let html = `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <div style="background:${{color}}33;border:1px solid ${{color}};border-radius:8px;padding:6px 14px;font-size:16px;font-weight:700;color:${{color}}">${{nd.op}}</div>
    </div>
    <div class="info-row"><div class="ik">Node Name</div><div class="iv">${{nd.name}}</div></div>
    <div class="info-row"><div class="ik">Op Type</div><div class="iv">${{nd.op}}</div></div>
    <div class="info-row"><div class="ik">Category</div><div class="iv">${{categoryLabel}}</div></div>
  `;

  // Input Tensors - Display input tensor names and shapes
  if(nd.input_shapes && nd.input_shapes.length) {{
    html += `<div class="section-title">Input Tensors</div>`;
    nd.input_shapes.forEach(t => {{
      html += `<div class="tensor-item">
        <div style="color:var(--accent);font-weight:600">${{t.name}}</div>
        <div style="color:var(--text2);margin-top:2px">${{t.shape}}</div>
      </div>`;
    }});
  }} else if(nd.inputs && nd.inputs.length) {{
    // If no shape info, at least show tensor names
    html += `<div class="section-title">Input Tensors</div>`;
    nd.inputs.forEach(name => {{
      html += `<div class="tensor-item">
        <div style="color:var(--accent);font-weight:600">${{name}}</div>
        <div style="color:var(--text2);margin-top:2px">Shape unknown</div>
      </div>`;
    }});
  }}

  // Output Tensors - Display output tensor names and shapes
  if(nd.output_shapes && nd.output_shapes.length) {{
    html += `<div class="section-title">Output Tensors</div>`;
    nd.output_shapes.forEach(t => {{
      html += `<div class="tensor-item">
        <div style="color:var(--accent);font-weight:600">${{t.name}}</div>
        <div style="color:var(--text2);margin-top:2px">${{t.shape}}</div>
      </div>`;
    }});
  }} else if(nd.outputs && nd.outputs.length) {{
    // If no shape info, at least show tensor names
    html += `<div class="section-title">Output Tensors</div>`;
    nd.outputs.forEach(name => {{
      html += `<div class="tensor-item">
        <div style="color:var(--accent);font-weight:600">${{name}}</div>
        <div style="color:var(--text2);margin-top:2px">Shape unknown</div>
      </div>`;
    }});
  }}

  // Attributes
  if(nd.attrs && Object.keys(nd.attrs).length) {{
    html += `<div class="section-title">Attributes</div>`;
    Object.entries(nd.attrs).forEach(([k,v]) => {{
      html += `<div class="attr-item"><div class="attr-key">${{k}}</div><div class="attr-val">${{JSON.stringify(v)}}</div></div>`;
    }});
  }}

  document.getElementById('graph-node-detail').innerHTML = html;
}}

// ── Pan & Zoom ──
let isPanning = false, panStart = {{x:0, y:0}}, panOrigin = {{x:0, y:0}};
const wrap = document.getElementById('canvas-wrap');

wrap.addEventListener('mousedown', e => {{
  if(e.target === svg || e.target === root || e.target.tagName === 'svg') {{
    isPanning = true;
    panStart = {{x: e.clientX, y: e.clientY}};
    panOrigin = {{x: transform.x, y: transform.y}};
    wrap.classList.add('grabbing');
  }}
}});
window.addEventListener('mousemove', e => {{
  if(!isPanning) return;
  transform.x = panOrigin.x + (e.clientX - panStart.x);
  transform.y = panOrigin.y + (e.clientY - panStart.y);
  applyTransform();
}});
window.addEventListener('mouseup', () => {{ isPanning = false; wrap.classList.remove('grabbing'); }});

wrap.addEventListener('wheel', e => {{
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  const rect = svg.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  transform.x = mx - (mx - transform.x) * factor;
  transform.y = my - (my - transform.y) * factor;
  transform.scale *= factor;
  applyTransform();
}}, {{passive: false}});

function zoomBy(f) {{
  const rect = svg.getBoundingClientRect();
  const cx = rect.width/2, cy = rect.height/2;
  transform.x = cx - (cx - transform.x) * f;
  transform.y = cy - (cy - transform.y) * f;
  transform.scale *= f;
  applyTransform();
}}

function fitView() {{
  const rect = svg.getBoundingClientRect();
  const scaleX = rect.width / (graphW + 80);
  const scaleY = rect.height / (graphH + 80);
  transform.scale = Math.min(scaleX, scaleY, 1);
  transform.x = (rect.width - graphW * transform.scale) / 2;
  transform.y = (rect.height - graphH * transform.scale) / 2;
  applyTransform();
}}

// ── Search ──
document.getElementById('graph-search').addEventListener('input', function() {{
  const q = this.value.toLowerCase().trim();
  root.querySelectorAll('.node-group').forEach(g => {{
    const nodeId = parseInt(g.dataset.id);
    const nd = graphNodes.find(n => n.id === nodeId);
    if(!nd) return;
    if(!q) {{ g.style.opacity = '1'; return; }}
    const match = nd.op.toLowerCase().includes(q) || nd.name.toLowerCase().includes(q);
    g.style.opacity = match ? '1' : '0.15';
  }});
}});

let graphPanelVisible = false;
function toggleGraphPanel() {{
  graphPanelVisible = !graphPanelVisible;
  const panel = document.getElementById('graph-detail-panel');
  if(graphPanelVisible) {{
    panel.classList.add('visible');
  }} else {{
    panel.classList.remove('visible');
  }}
}}

// ═══════════════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════════════

function switchTab(t){{
  document.getElementById("tab-debug").classList.toggle("active", t==="debug");
  document.getElementById("tab-graph").classList.toggle("active", t==="graph");
  document.getElementById("tab-stats").classList.toggle("active", t==="stats");
  
  document.getElementById("list-panel").style.display = t==="debug"?"flex":"none";
  document.getElementById("detail-panel").style.display = t==="debug"?"block":"none";
  document.getElementById("stats-panel").classList.toggle("active", t==="stats");
  document.getElementById("graph-panel").classList.toggle("active", t==="graph");
  
  const graphDetailPanel = document.getElementById('graph-detail-panel');
  if(t === "graph" && graphPanelVisible) {{
    graphDetailPanel.classList.add('visible');
  }} else {{
    graphDetailPanel.classList.remove('visible');
  }}
  
  if(t==="stats") buildStats();
  if(t==="graph") {{ buildGraph(); buildLegend(); setTimeout(fitView, 50); }}
}}

// ── init ─────────────────────────────────────────────────────────────────────
buildList();
if(nodes.length>0) selectNode(0);
</script>
</body>
</html>"""


# ============================================================================
# Module 8: CLI - Command Line Entry Point
# ============================================================================

def get_model_structure(model_path: str) -> dict:
    """
    Extract model structure without running inference.
    Returns node information with attributes but without actual tensor values.
    """
    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[Warning] Shape inference failed: {e}")

    # Build a mapping from tensor name to shape/dtype info
    tensor_info_map = {}
    
    # Get info from value_info
    for value_info in model.graph.value_info:
        name = value_info.name
        shape = []
        dtype_str = "unknown"
        
        if value_info.type.tensor_type:
            tt = value_info.type.tensor_type
            # Get shape
            for dim in tt.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            # Get dtype
            if tt.elem_type:
                from onnx import TensorProto
                dtype_map = {
                    TensorProto.FLOAT: "float32",
                    TensorProto.DOUBLE: "float64",
                    TensorProto.INT32: "int32",
                    TensorProto.INT64: "int64",
                    TensorProto.UINT8: "uint8",
                    TensorProto.INT8: "int8",
                    TensorProto.UINT16: "uint16",
                    TensorProto.INT16: "int16",
                    TensorProto.BOOL: "bool",
                }
                dtype_str = dtype_map.get(tt.elem_type, f"type_{tt.elem_type}")
        
        tensor_info_map[name] = {"shape": shape, "dtype": dtype_str}
    
    # Get info from inputs
    initializer_names = {init.name for init in model.graph.initializer}
    for input_info in model.graph.input:
        if input_info.name not in initializer_names:
            name = input_info.name
            shape = []
            dtype_str = "unknown"
            
            if input_info.type.tensor_type:
                tt = input_info.type.tensor_type
                for dim in tt.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(-1)
                if tt.elem_type:
                    from onnx import TensorProto
                    dtype_map = {
                        TensorProto.FLOAT: "float32",
                        TensorProto.DOUBLE: "float64",
                        TensorProto.INT32: "int32",
                        TensorProto.INT64: "int64",
                        TensorProto.UINT8: "uint8",
                        TensorProto.INT8: "int8",
                        TensorProto.UINT16: "uint16",
                        TensorProto.INT16: "int16",
                        TensorProto.BOOL: "bool",
                    }
                    dtype_str = dtype_map.get(tt.elem_type, f"type_{tt.elem_type}")
            
            tensor_info_map[name] = {"shape": shape, "dtype": dtype_str}
    
    # Get info from outputs
    for output_info in model.graph.output:
        name = output_info.name
        shape = []
        dtype_str = "unknown"
        
        if output_info.type.tensor_type:
            tt = output_info.type.tensor_type
            for dim in tt.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            if tt.elem_type:
                from onnx import TensorProto
                dtype_map = {
                    TensorProto.FLOAT: "float32",
                    TensorProto.DOUBLE: "float64",
                    TensorProto.INT32: "int32",
                    TensorProto.INT64: "int64",
                    TensorProto.UINT8: "uint8",
                    TensorProto.INT8: "int8",
                    TensorProto.UINT16: "uint16",
                    TensorProto.INT16: "int16",
                    TensorProto.BOOL: "bool",
                }
                dtype_str = dtype_map.get(tt.elem_type, f"type_{tt.elem_type}")
        
        tensor_info_map[name] = {"shape": shape, "dtype": dtype_str}
    
    # Get info from initializers (weights)
    for init in model.graph.initializer:
        name = init.name
        shape = list(init.dims)
        dtype_str = "unknown"
        
        from onnx import TensorProto
        dtype_map = {
            TensorProto.FLOAT: "float32",
            TensorProto.DOUBLE: "float64",
            TensorProto.INT32: "int32",
            TensorProto.INT64: "int64",
            TensorProto.UINT8: "uint8",
            TensorProto.INT8: "int8",
            TensorProto.UINT16: "uint16",
            TensorProto.INT16: "int16",
            TensorProto.BOOL: "bool",
        }
        dtype_str = dtype_map.get(init.data_type, f"type_{init.data_type}")
        
        tensor_info_map[name] = {"shape": shape, "dtype": dtype_str}

    result = {}
    for idx, node in enumerate(model.graph.node):
        info = get_node_static_info(node, idx)
        node_id = info["node_id"]

        # Create entries for inputs/outputs with shape info from tensor_info_map
        inputs_data = {}
        for name in info["input_names"]:
            if name:
                if name in tensor_info_map:
                    inputs_data[name] = {
                        "available": False,
                        "shape": tensor_info_map[name]["shape"],
                        "dtype": tensor_info_map[name]["dtype"],
                    }
                else:
                    inputs_data[name] = {"available": False}

        outputs_data = {}
        for name in info["output_names"]:
            if name:
                if name in tensor_info_map:
                    outputs_data[name] = {
                        "available": False,
                        "shape": tensor_info_map[name]["shape"],
                        "dtype": tensor_info_map[name]["dtype"],
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


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Debugger – capture every node's actual tensor values and produce an HTML report."
    )
    parser.add_argument("model",  help="Path to the ONNX model (.onnx)")
    parser.add_argument("input",  nargs="?", default=None,
                        help="Path to the input data (.npy) - optional, if not provided only shows node attributes")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML path (default: <model>_debug.html)")
    parser.add_argument("--inspect", "-i", default=None, metavar="NODE_ID",
                        help="Also pretty-print details for a single node (requires input file)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Dump the raw debug result to <model>_debug.json")
    args = parser.parse_args()

    # ── Validate model file ──────────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        print(f"[Error] Model file not found: {args.model}")
        sys.exit(1)

    # ── Default output path ──────────────────────────────────────────────────
    out_html = args.output
    if out_html is None:
        base = os.path.splitext(args.model)[0]
        out_html = base + "_debug.html"

    # ── Mode 1: Model-only mode (no input file) ────────────────────────────────
    if args.input is None:
        print(f"[*] Loading model  : {args.model}")
        print("[*] No input file provided - generating structure-only report")

        result = get_model_structure(args.model)

        n_nodes = len(result)
        print(f"[OK] Extracted {n_nodes} nodes (attributes only, no tensor values)")

        # ── Optional: Export JSON ──────────────────────────────────────────────────
        if args.json:
            json_path = os.path.splitext(args.model)[0] + "_debug.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"[*] JSON dump      : {json_path}")

        # ── Generate HTML report ─────────────────────────────────────────────────
        print("[*] Generating HTML report...")
        build_html(result, args.model, None, out_html)
        print(f"\n[OK] Open in browser: {os.path.abspath(out_html)}")

    # ── Mode 2: Full debug mode (with input file) ───────────────────────────────
    else:
        if not os.path.isfile(args.input):
            print(f"[Error] Input file not found: {args.input}")
            sys.exit(1)

        print(f"[*] Loading model  : {args.model}")
        print(f"[*] Input data     : {args.input}")

        debugger = OnnxDebugger(args.model)

        print("[*] Running inference (all intermediate outputs enabled)...")
        result = debugger.run(args.input)

        n_nodes   = len(result)
        n_tensors = sum(len(v["inputs"]) + len(v["outputs"]) for v in result.values())
        print(f"[OK] Captured {n_nodes} nodes, {n_tensors} tensor snapshots")

        # ── Optional: Inspect single node ────────────────────────────────────────
        if args.inspect:
            debugger.inspect_node(args.inspect, args.input)

        # ── Optional: Export JSON ──────────────────────────────────────────────────
        if args.json:
            json_path = os.path.splitext(args.model)[0] + "_debug.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"[*] JSON dump      : {json_path}")

        # ── Generate HTML report ─────────────────────────────────────────────────
        print("[*] Generating HTML report...")
        build_html(result, args.model, args.input, out_html)
        print(f"\n[OK] Open in browser: {os.path.abspath(out_html)}")


if __name__ == "__main__":
    main()
