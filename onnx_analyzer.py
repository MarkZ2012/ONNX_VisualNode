#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Model Analyzer
====================
解析ONNX文件，导出包含交互式图形可视化、算子统计和算力预估的HTML报告。
依赖：onnx, numpy（均已在pip list中）

用法：
    python onnx_analyzer.py model.onnx
    python onnx_analyzer.py model.onnx --output report.html
"""

import sys
import os
import json
import argparse
import collections
from datetime import datetime

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


# ─────────────────────────────────────────────
# 数据类型映射
# ─────────────────────────────────────────────
DTYPE_MAP = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
    11: "float64", 12: "uint32", 13: "uint64", 14: "complex64", 15: "complex128",
}

DTYPE_BYTES = {
    "float32": 4, "uint8": 1, "int8": 1, "uint16": 2, "int16": 2,
    "int32": 4, "int64": 8, "bool": 1, "float16": 2, "float64": 8,
    "uint32": 4, "uint64": 8, "complex64": 8, "complex128": 16,
}

# 算子类别颜色
OP_CATEGORY = {
    "Conv": "conv", "ConvTranspose": "conv",
    "Gemm": "gemm", "MatMul": "gemm",
    "Relu": "act", "Sigmoid": "act", "Tanh": "act", "LeakyRelu": "act",
    "Elu": "act", "Selu": "act", "Softmax": "act", "HardSigmoid": "act",
    "Mish": "act", "Swish": "act", "Gelu": "act", "PRelu": "act",
    "MaxPool": "pool", "AveragePool": "pool", "GlobalAveragePool": "pool",
    "GlobalMaxPool": "pool", "LpPool": "pool",
    "BatchNormalization": "norm", "LayerNormalization": "norm",
    "InstanceNormalization": "norm", "GroupNormalization": "norm",
    "LRN": "norm",
    "Add": "eltwise", "Sub": "eltwise", "Mul": "eltwise", "Div": "eltwise",
    "Sum": "eltwise", "Max": "eltwise", "Min": "eltwise", "Pow": "eltwise",
    "Reshape": "shape", "Flatten": "shape", "Squeeze": "shape",
    "Unsqueeze": "shape", "Transpose": "shape", "Concat": "shape",
    "Slice": "shape", "Gather": "shape", "GatherElements": "shape",
    "Expand": "shape", "Tile": "shape", "Pad": "shape",
    "Resize": "upsample", "Upsample": "upsample",
    "LSTM": "rnn", "GRU": "rnn", "RNN": "rnn",
    "Dropout": "other", "Identity": "other", "Constant": "other",
    "Shape": "other", "Cast": "other", "Clip": "other",
    "ReduceMean": "reduce", "ReduceSum": "reduce", "ReduceMax": "reduce",
    "ReduceMin": "reduce", "ReduceProd": "reduce", "ReduceL2": "reduce",
    "Attention": "attention", "MultiHeadAttention": "attention",
}


def get_op_category(op_type):
    return OP_CATEGORY.get(op_type, "other")


# ─────────────────────────────────────────────
# 形状推断工具
# ─────────────────────────────────────────────
def get_tensor_shape(type_proto):
    if type_proto.HasField("tensor_type"):
        shape = type_proto.tensor_type.shape
        if shape:
            dims = []
            for d in shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
            return dims
    return None


def shape_to_str(shape):
    if shape is None:
        return "unknown"
    return "[" + ", ".join(str(d) for d in shape) + "]"


def calc_elements(shape):
    if shape is None:
        return None
    total = 1
    for d in shape:
        if isinstance(d, int) and d > 0:
            total *= d
        else:
            return None
    return total


# ─────────────────────────────────────────────
# 算力估算
# ─────────────────────────────────────────────
def estimate_flops(node, value_info_map, initializer_shapes):
    """粗略估算节点的FLOPs"""
    op = node.op_type
    try:
        if op in ("Conv", "ConvTranspose"):
            # FLOPs = 2 * Cin * Kh * Kw * Oh * Ow * Cout
            output_name = node.output[0] if node.output else None
            input_name = node.input[0] if node.input else None
            weight_name = node.input[1] if len(node.input) > 1 else None

            out_shape = value_info_map.get(output_name)
            in_shape = value_info_map.get(input_name)
            w_shape = initializer_shapes.get(weight_name)

            if out_shape and w_shape and len(out_shape) >= 4 and len(w_shape) >= 4:
                N = out_shape[0] if isinstance(out_shape[0], int) and out_shape[0] > 0 else 1
                Cout = out_shape[1] if isinstance(out_shape[1], int) else w_shape[0]
                Oh = out_shape[2] if isinstance(out_shape[2], int) else 1
                Ow = out_shape[3] if isinstance(out_shape[3], int) else 1
                Cin = w_shape[1]
                Kh = w_shape[2]
                Kw = w_shape[3]
                # groups
                groups = 1
                for attr in node.attribute:
                    if attr.name == "group":
                        groups = attr.i
                if isinstance(Cin, int) and isinstance(Cout, int):
                    flops = 2 * N * (Cin * groups) * Kh * Kw * Oh * Ow * Cout // groups
                    return int(flops)

        elif op in ("Gemm", "MatMul"):
            # FLOPs = 2 * M * N * K
            if op == "Gemm":
                a_name = node.input[0] if node.input else None
                b_name = node.input[1] if len(node.input) > 1 else None
                a_shape = value_info_map.get(a_name) or initializer_shapes.get(a_name)
                b_shape = value_info_map.get(b_name) or initializer_shapes.get(b_name)
                if a_shape and b_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
                    M = a_shape[0]; K = a_shape[1]
                    N = b_shape[1]
                    if all(isinstance(x, int) and x > 0 for x in [M, K, N]):
                        return int(2 * M * K * N)
            else:
                a_name = node.input[0] if node.input else None
                b_name = node.input[1] if len(node.input) > 1 else None
                a_shape = value_info_map.get(a_name) or initializer_shapes.get(a_name)
                b_shape = value_info_map.get(b_name) or initializer_shapes.get(b_name)
                if a_shape and b_shape:
                    if len(a_shape) >= 2 and len(b_shape) >= 2:
                        K = a_shape[-1]; N = b_shape[-1]
                        M = a_shape[-2] if len(a_shape) >= 2 else 1
                        if all(isinstance(x, int) and x > 0 for x in [M, K, N]):
                            return int(2 * M * K * N)

        elif op in ("MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"):
            output_name = node.output[0] if node.output else None
            out_shape = value_info_map.get(output_name)
            if out_shape and len(out_shape) >= 4:
                kh, kw = 1, 1
                for attr in node.attribute:
                    if attr.name == "kernel_shape" and len(attr.ints) >= 2:
                        kh, kw = attr.ints[0], attr.ints[1]
                N = out_shape[0] if isinstance(out_shape[0], int) and out_shape[0] > 0 else 1
                C = out_shape[1] if isinstance(out_shape[1], int) else 1
                Oh = out_shape[2] if isinstance(out_shape[2], int) else 1
                Ow = out_shape[3] if isinstance(out_shape[3], int) else 1
                if isinstance(C, int):
                    return int(N * C * Oh * Ow * kh * kw)

        elif op == "BatchNormalization":
            input_name = node.input[0] if node.input else None
            in_shape = value_info_map.get(input_name)
            if in_shape:
                elems = calc_elements(in_shape)
                if elems:
                    return int(elems * 4)  # rough: sub,mul,add,scale

    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────
# 主解析函数
# ─────────────────────────────────────────────
def analyze_onnx(model_path):
    model = onnx.load(model_path)

    # 尝试形状推断
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph

    # 收集value_info（张量形状）
    value_info_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        shape = get_tensor_shape(vi.type)
        if shape is not None:
            value_info_map[vi.name] = shape

    # 收集initializer形状
    initializer_shapes = {}
    initializer_set = set()
    for init in graph.initializer:
        initializer_set.add(init.name)
        initializer_shapes[init.name] = list(init.dims)

    # 模型输入输出
    model_inputs = []
    for inp in graph.input:
        if inp.name not in initializer_set:
            shape = get_tensor_shape(inp.type)
            dtype_id = inp.type.tensor_type.elem_type
            model_inputs.append({
                "name": inp.name,
                "shape": shape_to_str(shape),
                "dtype": DTYPE_MAP.get(dtype_id, "unknown"),
            })

    model_outputs = []
    for out in graph.output:
        shape = get_tensor_shape(out.type)
        dtype_id = out.type.tensor_type.elem_type
        model_outputs.append({
            "name": out.name,
            "shape": shape_to_str(shape),
            "dtype": DTYPE_MAP.get(dtype_id, "unknown"),
        })

    # 节点解析
    # 节点布局：先放 Input 节点，再放算子节点，最后放 Output 节点
    nodes_data = []
    op_counter = collections.Counter()
    total_flops = 0
    total_params = 0

    # 计算参数量
    for init in graph.initializer:
        elems = calc_elements(list(init.dims))
        if elems:
            total_params += elems

    # ── 第一步：为每个模型输入创建 Input 节点 ──
    # 使用负数 id 区分（前端用 node_id 字段，不用数组下标）
    input_node_map = {}   # tensor_name -> node_id in nodes_data
    for inp in graph.input:
        if inp.name in initializer_set:
            continue  # 权重，不作为图节点
        shape = get_tensor_shape(inp.type)
        dtype_id = inp.type.tensor_type.elem_type
        nid = len(nodes_data)
        input_node_map[inp.name] = nid
        nodes_data.append({
            "id": nid,
            "name": inp.name,
            "op": "Input",
            "category": "input",
            "inputs": [],
            "outputs": [inp.name],
            "input_shapes": [],
            "output_shapes": [{"name": inp.name[:40],
                                "shape": shape_to_str(shape)}],
            "attrs": {"dtype": DTYPE_MAP.get(dtype_id, "unknown"),
                      "shape": shape_to_str(shape)},
            "flops": 0,
        })

    # ── 第二步：建立 tensor_name -> 算子节点索引映射（基于 op 节点输出）──
    # 偏移量：算子节点 id = len(input_node_map) + graph_node_index
    op_id_offset = len(nodes_data)
    output_to_node_id = {}  # tensor_name -> node_id in nodes_data
    for idx, node in enumerate(graph.node):
        nid = op_id_offset + idx
        for out in node.output:
            if out:
                output_to_node_id[out] = nid
    # 也把模型 input 张量映射过来
    for tname, nid in input_node_map.items():
        output_to_node_id[tname] = nid

    # ── 第三步：解析算子节点 ──
    for idx, node in enumerate(graph.node):
        nid = op_id_offset + idx
        op = node.op_type
        op_counter[op] += 1

        flops = estimate_flops(node, value_info_map, initializer_shapes)
        total_flops += flops

        # 属性
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = round(attr.f, 6)
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode("utf-8", errors="replace")
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = [round(f, 6) for f in attr.floats]

        # 输入输出形状（排除 initializer 权重的连线显示，但保留形状信息）
        input_shapes = []
        for inp_name in node.input:
            if inp_name:
                sh = value_info_map.get(inp_name) or initializer_shapes.get(inp_name)
                input_shapes.append({"name": inp_name[:40], "shape": shape_to_str(sh)})

        output_shapes = []
        for out_name in node.output:
            if out_name:
                sh = value_info_map.get(out_name)
                output_shapes.append({"name": out_name[:40], "shape": shape_to_str(sh)})

        nodes_data.append({
            "id": nid,
            "name": node.name or f"{op}_{idx}",
            "op": op,
            "category": get_op_category(op),
            "inputs": [n for n in node.input if n],
            "outputs": [n for n in node.output if n],
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "attrs": attrs,
            "flops": flops,
        })

    # ── 第四步：为每个模型输出创建 Output 节点 ──
    output_node_ids = []
    for out in graph.output:
        shape = get_tensor_shape(out.type)
        dtype_id = out.type.tensor_type.elem_type
        nid = len(nodes_data)
        output_node_ids.append(nid)
        nodes_data.append({
            "id": nid,
            "name": out.name,
            "op": "Output",
            "category": "output",
            "inputs": [out.name],
            "outputs": [],
            "input_shapes": [{"name": out.name[:40],
                               "shape": shape_to_str(shape)}],
            "output_shapes": [],
            "attrs": {"dtype": DTYPE_MAP.get(dtype_id, "unknown"),
                      "shape": shape_to_str(shape)},
            "flops": 0,
        })
        # 注册 output 张量名到 output_to_node_id（作为 dst）
        # output tensor 节点接收来自算子的张量
        # 这里不需要注册到 output_to_node_id，只用于建边 dst

    # ── 第五步：构建边（基于 tensor name 精确匹配）──
    edges = []
    seen_edges = set()

    def add_edge(src_id, dst_id, tensor_name):
        key = (src_id, dst_id, tensor_name)
        if key not in seen_edges and src_id != dst_id:
            seen_edges.add(key)
            edges.append({"src": src_id, "dst": dst_id, "tensor": tensor_name[:40]})

    # 算子节点的输入边
    for idx, node in enumerate(graph.node):
        nid = op_id_offset + idx
        for inp_name in node.input:
            if not inp_name:
                continue
            if inp_name in initializer_set:
                continue  # 跳过权重，不画权重连线
            src_id = output_to_node_id.get(inp_name)
            if src_id is not None:
                add_edge(src_id, nid, inp_name)

    # 模型输出节点的输入边
    for out_node_idx, out in enumerate(graph.output):
        dst_id = output_node_ids[out_node_idx]
        src_id = output_to_node_id.get(out.name)
        if src_id is not None:
            add_edge(src_id, dst_id, out.name)

    # 算子统计
    op_stats = []
    for op, count in op_counter.most_common():
        op_stats.append({
            "op": op,
            "count": count,
            "category": get_op_category(op),
        })

    # 模型元信息
    meta = {
        "model_path": os.path.basename(model_path),
        "ir_version": model.ir_version,
        "opset": [{"domain": op.domain or "ai.onnx", "version": op.version}
                  for op in model.opset_import],
        "doc_string": model.doc_string[:200] if model.doc_string else "",
        "producer": model.producer_name,
        "model_version": model.model_version,
        "total_nodes": len(graph.node),
        "total_params": total_params,
        "total_flops": total_flops,
        "inputs": model_inputs,
        "outputs": model_outputs,
        "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return meta, nodes_data, edges, op_stats


# ─────────────────────────────────────────────
# 格式化数字
# ─────────────────────────────────────────────
def fmt_num(n):
    if n >= 1e12:
        return f"{n/1e12:.2f} T"
    if n >= 1e9:
        return f"{n/1e9:.2f} G"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.2f} K"
    return str(n)


# ─────────────────────────────────────────────
# HTML生成
# ─────────────────────────────────────────────
def generate_html(meta, nodes_data, edges, op_stats, output_path):
    data_json = json.dumps({
        "meta": meta,
        "nodes": nodes_data,
        "edges": edges,
        "op_stats": op_stats,
    }, ensure_ascii=False, separators=(",", ":"))

    total_nodes = meta["total_nodes"]
    total_params_str = fmt_num(meta["total_params"])
    total_flops_str = fmt_num(meta["total_flops"])
    model_name = meta["model_path"]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ONNX Analyzer - {model_name}</title>
<style>
:root {{
  --bg: #0f1117;
  --panel: #1a1d27;
  --panel2: #21253a;
  --border: #2d3250;
  --accent: #4f8ef7;
  --accent2: #7c4dff;
  --text: #e2e8f0;
  --text2: #8892a4;
  --success: #22c55e;
  --warn: #f59e0b;
  --danger: #ef4444;

  --c-conv: #4f8ef7;
  --c-gemm: #7c4dff;
  --c-act: #22c55e;
  --c-pool: #06b6d4;
  --c-norm: #f59e0b;
  --c-eltwise: #ec4899;
  --c-shape: #8b5cf6;
  --c-upsample: #10b981;
  --c-rnn: #f97316;
  --c-reduce: #84cc16;
  --c-attention: #e11d48;
  --c-other: #6b7280;
  --c-input: #00d4aa;
  --c-output: #ff6b6b;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

/* ── Header ── */
header {{ background: var(--panel); border-bottom: 1px solid var(--border); padding: 10px 20px; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
.logo {{ font-size: 20px; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; white-space: nowrap; }}
.model-name {{ font-size: 14px; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }}
.header-stats {{ display: flex; gap: 20px; flex-shrink: 0; }}
.stat-chip {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 8px; padding: 4px 12px; font-size: 12px; }}
.stat-chip span {{ color: var(--accent); font-weight: 700; }}
.tab-bar {{ display: flex; gap: 4px; flex-shrink: 0; }}
.tab {{ padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; color: var(--text2); border: 1px solid transparent; transition: all .2s; }}
.tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.tab:hover:not(.active) {{ background: var(--panel2); color: var(--text); }}

/* ── Main layout ── */
.main {{ display: flex; flex: 1; overflow: hidden; }}

/* ── Graph canvas ── */
#graph-panel {{ flex: 1; position: relative; overflow: hidden; background: var(--bg); }}
#canvas-wrap {{ width: 100%; height: 100%; cursor: grab; }}
#canvas-wrap.grabbing {{ cursor: grabbing; }}
svg#graph-svg {{ width: 100%; height: 100%; }}
.minimap {{ position: absolute; bottom: 16px; right: 16px; background: rgba(26,29,39,0.9); border: 1px solid var(--border); border-radius: 8px; width: 160px; height: 100px; overflow: hidden; }}
.controls {{ position: absolute; bottom: 16px; left: 16px; display: flex; flex-direction: column; gap: 6px; }}
.ctrl-btn {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px; color: var(--text); width: 32px; height: 32px; cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center; transition: background .2s; }}
.ctrl-btn:hover {{ background: var(--panel2); }}
.search-box {{ position: absolute; top: 12px; left: 12px; }}
#node-search {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 6px 12px; font-size: 13px; width: 220px; outline: none; }}
#node-search:focus {{ border-color: var(--accent); }}

/* ── SVG nodes ── */
.node-group {{ cursor: pointer; transition: filter .15s; }}
.node-group:hover {{ filter: brightness(1.3); }}
.node-rect {{ rx: 8; ry: 8; stroke-width: 1.5; }}
.node-op {{ font-size: 12px; font-weight: 700; fill: #fff; text-anchor: middle; dominant-baseline: central; pointer-events: none; }}
.node-name {{ font-size: 9px; fill: rgba(255,255,255,0.6); text-anchor: middle; pointer-events: none; }}
.edge-path {{ fill: none; stroke: #3a3f5c; stroke-width: 1.2; marker-end: url(#arrow); opacity: 0.6; }}
.edge-path.highlighted {{ stroke: var(--accent); opacity: 1; stroke-width: 2; }}
.node-rect.selected {{ stroke: #fff; stroke-width: 3; filter: drop-shadow(0 0 8px var(--accent)); }}

/* ── Right panel ── */
#right-panel {{ width: 320px; background: var(--panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden; }}
#right-panel.hidden {{ display: none; }}
.panel-header {{ padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 600; color: var(--text2); display: flex; align-items: center; justify-content: space-between; }}
.panel-content {{ flex: 1; overflow-y: auto; padding: 12px; }}
.panel-content::-webkit-scrollbar {{ width: 4px; }}
.panel-content::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

.info-row {{ display: flex; gap: 8px; margin-bottom: 8px; font-size: 12px; flex-wrap: wrap; }}
.info-key {{ color: var(--text2); width: 80px; flex-shrink: 0; }}
.info-val {{ color: var(--text); word-break: break-all; flex: 1; }}
.section-title {{ font-size: 11px; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; margin: 12px 0 6px; }}
.tensor-item {{ background: var(--panel2); border-radius: 6px; padding: 6px 10px; margin-bottom: 4px; font-size: 11px; }}
.tensor-name {{ color: var(--text); word-break: break-all; }}
.tensor-shape {{ color: var(--text2); margin-top: 2px; }}
.attr-item {{ display: flex; gap: 6px; font-size: 11px; margin-bottom: 4px; }}
.attr-key {{ color: var(--warn); min-width: 80px; flex-shrink: 0; }}
.attr-val {{ color: var(--text); word-break: break-all; }}
.flops-badge {{ display: inline-block; background: linear-gradient(135deg, var(--accent), var(--accent2)); border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; color: #fff; margin-top: 4px; }}
.no-select {{ color: var(--text2); font-size: 13px; text-align: center; padding: 40px 20px; }}

/* ── Stats panel ── */
#stats-panel {{ display: none; flex: 1; overflow: hidden; flex-direction: column; }}
#stats-panel.active {{ display: flex; }}
.stats-content {{ flex: 1; overflow-y: auto; padding: 16px; }}
.stats-content::-webkit-scrollbar {{ width: 6px; }}
.stats-content::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.s-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
.s-card-label {{ font-size: 12px; color: var(--text2); margin-bottom: 8px; }}
.s-card-value {{ font-size: 28px; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.s-card-sub {{ font-size: 11px; color: var(--text2); margin-top: 4px; }}

.op-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.op-table th {{ text-align: left; padding: 8px 12px; font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }}
.op-table td {{ padding: 8px 12px; border-bottom: 1px solid rgba(45,50,80,0.5); }}
.op-table tr:hover td {{ background: var(--panel2); }}
.op-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }}
.bar-cell {{ min-width: 100px; }}
.bar-bg {{ background: var(--panel2); border-radius: 4px; height: 6px; }}
.bar-fill {{ height: 6px; border-radius: 4px; }}

.io-section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
.io-title {{ font-size: 13px; font-weight: 600; margin-bottom: 12px; color: var(--text2); }}
.io-item {{ background: var(--panel2); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 12px; }}
.io-name {{ color: var(--accent); word-break: break-all; margin-bottom: 4px; font-weight: 600; }}
.io-meta {{ color: var(--text2); }}

/* ── Tooltip ── */
#tooltip {{ position: fixed; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 12px; pointer-events: none; z-index: 9999; display: none; max-width: 240px; line-height: 1.5; box-shadow: 0 8px 24px rgba(0,0,0,0.4); }}

/* ── Legend ── */
.legend {{ position: absolute; top: 12px; right: 12px; background: rgba(26,29,39,0.92); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 11px; display: flex; flex-direction: column; gap: 4px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }}
</style>
</head>
<body>

<header>
  <div class="logo">⬡ ONNX Analyzer</div>
  <div class="model-name" title="{model_name}">{model_name}</div>
  <div class="header-stats">
    <div class="stat-chip">节点 <span>{total_nodes}</span></div>
    <div class="stat-chip">参数 <span>{total_params_str}</span></div>
    <div class="stat-chip">FLOPs <span>{total_flops_str}</span></div>
  </div>
  <div class="tab-bar">
    <div class="tab active" id="tab-graph" onclick="switchTab('graph')">图形视图</div>
    <div class="tab" id="tab-stats" onclick="switchTab('stats')">统计分析</div>
  </div>
</header>

<div class="main">

  <!-- ── Graph Panel ── -->
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

    <div class="search-box">
      <input id="node-search" type="text" placeholder="🔍 搜索算子或节点...">
    </div>

    <div class="controls">
      <button class="ctrl-btn" title="放大" onclick="zoomBy(1.25)">+</button>
      <button class="ctrl-btn" title="缩小" onclick="zoomBy(0.8)">−</button>
      <button class="ctrl-btn" title="适应屏幕" onclick="fitView()" style="font-size:13px">⛶</button>
      <button class="ctrl-btn" title="切换信息面板" onclick="togglePanel()" style="font-size:13px">☰</button>
    </div>

    <div class="legend" id="legend-panel"></div>
  </div>

  <!-- ── Right Info Panel ── -->
  <div id="right-panel">
    <div class="panel-header">
      <span>节点详情</span>
      <span style="font-size:10px;cursor:pointer;color:var(--text2)" onclick="togglePanel()">✕</span>
    </div>
    <div class="panel-content" id="node-detail">
      <div class="no-select">点击图中节点查看详情</div>
    </div>
  </div>

  <!-- ── Stats Panel ── -->
  <div id="stats-panel">
    <div class="stats-content" id="stats-content"></div>
  </div>

</div>

<div id="tooltip"></div>

<script>
const RAW = {data_json};
const meta = RAW.meta;
const nodes = RAW.nodes;
const edges = RAW.edges;
const opStats = RAW.op_stats;

// ── Category colors ──
const CAT_COLOR = {{
  conv:"#4f8ef7", gemm:"#7c4dff", act:"#22c55e", pool:"#06b6d4",
  norm:"#f59e0b", eltwise:"#ec4899", shape:"#8b5cf6", upsample:"#10b981",
  rnn:"#f97316", reduce:"#84cc16", attention:"#e11d48", other:"#6b7280",
  input:"#00d4aa", output:"#ff6b6b"
}};
const CAT_LABEL = {{
  conv:"卷积", gemm:"全连接", act:"激活", pool:"池化",
  norm:"归一化", eltwise:"逐元素", shape:"形状", upsample:"上采样",
  rnn:"循环", reduce:"归约", attention:"注意力", other:"其他",
  input:"输入", output:"输出"
}};

function opColor(cat) {{ return CAT_COLOR[cat] || "#6b7280"; }}

// ── Layout ──
const NODE_W = 130, NODE_H = 46, LEVEL_GAP = 80, COL_GAP = 150;
const IO_W = 160, IO_H = 36;  // Input/Output nodes are wider, shorter

function getNodeDims(nd) {{
  if(nd.category === 'input' || nd.category === 'output') return [IO_W, IO_H];
  return [NODE_W, NODE_H];
}}

function computeLayout(nodes, edges) {{
  const n = nodes.length;
  if(n === 0) return;

  // Build id->index map since node ids may not be 0..n-1 sequential
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
    const [w, h] = getNodeDims(nd);
    nd._w = w; nd._h = h;
    nd._level = lv;   // store level for bypass routing
    nd.y = lv * (NODE_H + LEVEL_GAP) + 40;
    nd.x = (posInGrp - (totalInGrp-1)/2) * (NODE_W + COL_GAP);
  }});

  // Expose levelGroups so edge routing can query nodes at each level
  nodes._levelGroups = levelGroups;
  nodes._idToIdx = idToIdx;
  nodes._level = level;
}}

computeLayout(nodes, edges);

// center offset
let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
nodes.forEach(nd => {{
  if(nd.x !== undefined) {{
    const [w,h] = getNodeDims(nd);
    minX = Math.min(minX, nd.x); minY = Math.min(minY, nd.y);
    maxX = Math.max(maxX, nd.x + w); maxY = Math.max(maxY, nd.y + h);
  }}
}});
const graphW = maxX - minX + NODE_W*2;
const graphH = maxY - minY + NODE_H*2;
const offsetX = -minX + NODE_W;
const offsetY = -minY + NODE_H;
nodes.forEach(nd => {{ if(nd.x !== undefined) {{ nd.x += offsetX; nd.y += offsetY; }} }});

// ── SVG rendering ──
const svg = document.getElementById('graph-svg');
const root = document.getElementById('graph-root');
let transform = {{ x: 0, y: 0, scale: 1 }};

function applyTransform() {{
  root.setAttribute('transform', `translate(${{transform.x}},${{transform.y}}) scale(${{transform.scale}})`);
}}

function buildGraph() {{
  root.innerHTML = '';

  // Build id->node map
  const nodeById = {{}};
  nodes.forEach(nd => nodeById[nd.id] = nd);

  // ── Pre-compute per-node out-edges and in-edges for anchor assignment ──
  // outEdges[nodeId] = [edgeIndex, ...]  (ordered by destination x position)
  // inEdges[nodeId]  = [edgeIndex, ...]
  const outEdges = {{}};
  const inEdges  = {{}};
  edges.forEach((e, ei) => {{
    if(!outEdges[e.src]) outEdges[e.src] = [];
    if(!inEdges[e.dst])  inEdges[e.dst]  = [];
    outEdges[e.src].push(ei);
    inEdges[e.dst].push(ei);
  }});

  // Sort each node's out-edges left→right by destination center-x
  Object.keys(outEdges).forEach(srcId => {{
    outEdges[+srcId].sort((a, b) => {{
      const da = nodeById[edges[a].dst], db = nodeById[edges[b].dst];
      if(!da || !db) return 0;
      const [daw] = getNodeDims(da), [dbw] = getNodeDims(db);
      return (da.x + daw/2) - (db.x + dbw/2);
    }});
  }});
  // Sort in-edges left→right by source center-x
  Object.keys(inEdges).forEach(dstId => {{
    inEdges[+dstId].sort((a, b) => {{
      const sa = nodeById[edges[a].src], sb = nodeById[edges[b].src];
      if(!sa || !sb) return 0;
      const [saw] = getNodeDims(sa), [sbw] = getNodeDims(sb);
      return (sa.x + saw/2) - (sb.x + sbw/2);
    }});
  }});

  // Compute anchor x for an edge at a node's bottom (source) or top (dest)
  function anchorX(nodeId, edgeIdx, side) {{
    const nd = nodeById[nodeId];
    if(!nd) return 0;
    const [w] = getNodeDims(nd);
    const list = side === 'out' ? (outEdges[nodeId] || []) : (inEdges[nodeId] || []);
    const total = list.length;
    if(total <= 1) return nd.x + w / 2;
    const pos = list.indexOf(edgeIdx);
    const margin = w * 0.10;
    const span   = w - margin * 2;
    return nd.x + margin + (pos / (total - 1)) * span;
  }}

  // ── Bypass lane allocator ──
  // For edges that skip >1 level, route them along a vertical "bypass lane"
  // to the left or right of the main column, so they don't cut through nodes.
  //
  // We need to know the horizontal extents of all nodes at each intermediate
  // level so we can pick a lane that clears them.
  //
  // Lane map: bypassLanes[side]['left'|'right'] = current outermost x used
  // We assign a new lane per long edge, incrementing outward.
  const BYPASS_MARGIN = 18;   // gap between node edge and bypass line
  const BYPASS_STEP   = 14;   // extra spacing between stacked bypass lines
  const bypassLaneCountL = {{}};  // key=levelRange string → count used on left
  const bypassLaneCountR = {{}};

  function getLevelXBounds(fromLevel, toLevel) {{
    // Return {{minX, maxX}} of all nodes at levels between fromLevel+1 and toLevel-1 (exclusive)
    let minX = Infinity, maxX = -Infinity;
    for(let lv = fromLevel; lv <= toLevel; lv++) {{
      nodes.forEach(nd => {{
        if(nd._level === lv) {{
          const [w] = getNodeDims(nd);
          minX = Math.min(minX, nd.x);
          maxX = Math.max(maxX, nd.x + w);
        }}
      }});
    }}
    return {{ minX: isFinite(minX) ? minX : 0, maxX: isFinite(maxX) ? maxX : 0 }};
  }}

  // For each long edge, decide: go left or right?
  // If src anchor is on the left half of src node → go left, else right
  // Then find a clear lane x coordinate
  function bypassPath(e, ei, sx, sy, dx, dy) {{
    const src = nodeById[e.src], dst = nodeById[e.dst];
    const srcLv = src._level, dstLv = dst._level;
    const laneKey = `${{srcLv}}-${{dstLv}}`;

    const bounds = getLevelXBounds(srcLv, dstLv);

    // Decide side: steer toward whichever side the anchor is closer to
    const [sw] = getNodeDims(src);
    const srcMid = src.x + sw / 2;
    const goLeft = sx <= srcMid;

    if(goLeft) {{
      if(!bypassLaneCountL[laneKey]) bypassLaneCountL[laneKey] = 0;
      const laneIdx = bypassLaneCountL[laneKey]++;
      const lx = bounds.minX - BYPASS_MARGIN - laneIdx * BYPASS_STEP;
      // Path: down from src → left to lane → down → right to dst top
      const STUB = 20;  // short vertical stub before turning
      return `M${{sx}},${{sy}}` +
             ` L${{sx}},${{sy + STUB}}` +
             ` L${{lx}},${{sy + STUB}}` +
             ` L${{lx}},${{dy - STUB}}` +
             ` L${{dx}},${{dy - STUB}}` +
             ` L${{dx}},${{dy}}`;
    }} else {{
      if(!bypassLaneCountR[laneKey]) bypassLaneCountR[laneKey] = 0;
      const laneIdx = bypassLaneCountR[laneKey]++;
      const rx = bounds.maxX + BYPASS_MARGIN + laneIdx * BYPASS_STEP;
      const STUB = 20;
      return `M${{sx}},${{sy}}` +
             ` L${{sx}},${{sy + STUB}}` +
             ` L${{rx}},${{sy + STUB}}` +
             ` L${{rx}},${{dy - STUB}}` +
             ` L${{dx}},${{dy - STUB}}` +
             ` L${{dx}},${{dy}}`;
    }}
  }}

  // edges first
  const edgeG = document.createElementNS('http://www.w3.org/2000/svg','g');
  edgeG.id = 'edges-g';
  edges.forEach((e, ei) => {{
    const src = nodeById[e.src], dst = nodeById[e.dst];
    if(!src || !dst || src.x === undefined || dst.x === undefined) return;
    const [sw, sh] = getNodeDims(src);
    const [dw, dh] = getNodeDims(dst);

    const sx = anchorX(e.src, ei, 'out');
    const sy = src.y + sh;
    const dx = anchorX(e.dst, ei, 'in');
    const dy = dst.y;

    const levelSpan = (dst._level || 0) - (src._level || 0);
    let pathD;

    if(levelSpan > 1) {{
      // Long edge: use bypass routing to avoid crossing intermediate nodes
      pathD = bypassPath(e, ei, sx, sy, dx, dy);
    }} else {{
      // Short edge (adjacent levels): simple cubic bezier
      const cp = Math.max(Math.abs(dy - sy) * 0.5, 30);
      pathD = `M${{sx}},${{sy}} C${{sx}},${{sy + cp}} ${{dx}},${{dy - cp}} ${{dx}},${{dy}}`;
    }}

    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d', pathD);
    path.setAttribute('class','edge-path');
    path.setAttribute('data-src', e.src);
    path.setAttribute('data-dst', e.dst);
    path.setAttribute('data-ei', ei);
    edgeG.appendChild(path);
  }});
  root.appendChild(edgeG);

  // nodes
  const nodeG = document.createElementNS('http://www.w3.org/2000/svg','g');
  nodeG.id = 'nodes-g';
  nodes.forEach((nd, i) => {{
    if(nd.x === undefined) return;
    const [w, h] = getNodeDims(nd);
    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    g.setAttribute('class','node-group');
    g.setAttribute('data-id', nd.id);
    g.setAttribute('transform', `translate(${{nd.x}},${{nd.y}})`);

    const color = opColor(nd.category);
    const isIO = nd.category === 'input' || nd.category === 'output';

    if(isIO) {{
      // Netron-style: pill / rounded rectangle for I/O nodes
      const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
      rect.setAttribute('class','node-rect');
      rect.setAttribute('width', w);
      rect.setAttribute('height', h);
      rect.setAttribute('fill', color+'33');
      rect.setAttribute('stroke', color);
      rect.setAttribute('stroke-width', '2');
      rect.setAttribute('rx', h/2);  // fully rounded ends = pill shape
      g.appendChild(rect);

      // Label: "▶ Input" or "◀ Output" + tensor name
      const icon = nd.category === 'input' ? '▶' : '◀';
      const typeText = document.createElementNS('http://www.w3.org/2000/svg','text');
      typeText.setAttribute('x', w/2);
      typeText.setAttribute('y', 13);
      typeText.setAttribute('text-anchor','middle');
      typeText.setAttribute('dominant-baseline','central');
      typeText.setAttribute('font-size','10');
      typeText.setAttribute('font-weight','700');
      typeText.setAttribute('fill', color);
      typeText.textContent = `${{icon}} ${{nd.category === 'input' ? 'INPUT' : 'OUTPUT'}}`;
      g.appendChild(typeText);

      const nameText = document.createElementNS('http://www.w3.org/2000/svg','text');
      nameText.setAttribute('x', w/2);
      nameText.setAttribute('y', 26);
      nameText.setAttribute('text-anchor','middle');
      nameText.setAttribute('dominant-baseline','central');
      nameText.setAttribute('font-size','9');
      nameText.setAttribute('fill','rgba(255,255,255,0.7)');
      const dispName = nd.name.length > 22 ? nd.name.slice(0,21)+'…' : nd.name;
      nameText.textContent = dispName;
      g.appendChild(nameText);
    }} else {{
      // Regular op node
      const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
      rect.setAttribute('class','node-rect');
      rect.setAttribute('width', w);
      rect.setAttribute('height', h);
      rect.setAttribute('fill', color+'22');
      rect.setAttribute('stroke', color);
      rect.setAttribute('rx', 8);
      g.appendChild(rect);

      const opText = document.createElementNS('http://www.w3.org/2000/svg','text');
      opText.setAttribute('class','node-op');
      opText.setAttribute('x', w/2);
      opText.setAttribute('y', 18);
      opText.textContent = nd.op.length > 15 ? nd.op.slice(0,14)+'…' : nd.op;
      opText.setAttribute('fill', color);
      g.appendChild(opText);

      const nameText = document.createElementNS('http://www.w3.org/2000/svg','text');
      nameText.setAttribute('class','node-name');
      nameText.setAttribute('x', w/2);
      nameText.setAttribute('y', 36);
      const dispName = nd.name.length > 18 ? nd.name.slice(0,17)+'…' : nd.name;
      nameText.textContent = dispName;
      g.appendChild(nameText);
    }}

    g.addEventListener('click', () => selectNode(nd.id));
    g.addEventListener('mouseenter', (ev) => showTooltip(ev, nd));
    g.addEventListener('mouseleave', hideTooltip);
    nodeG.appendChild(g);
  }});
  root.appendChild(nodeG);
}}

// ── Legend ──
function buildLegend() {{
  const usedCats = [...new Set(nodes.map(n => n.category))];
  const leg = document.getElementById('legend-panel');
  leg.innerHTML = usedCats.map(c =>
    `<div class="legend-item"><div class="legend-dot" style="background:${{opColor(c)}}"></div><span>${{CAT_LABEL[c]||c}}</span></div>`
  ).join('');
}}

// ── Node selection ──
let selectedNode = null;
function selectNode(nodeId) {{
  // deselect previous
  if(selectedNode !== null) {{
    const prev = root.querySelector(`[data-id="${{selectedNode}}"] .node-rect`);
    if(prev) prev.classList.remove('selected');
  }}
  // highlight edges
  root.querySelectorAll('.edge-path').forEach(p => {{
    p.classList.remove('highlighted');
    p.setAttribute('marker-end','url(#arrow)');
  }});

  selectedNode = nodeId;
  const el = root.querySelector(`[data-id="${{nodeId}}"] .node-rect`);
  if(el) el.classList.add('selected');

  // highlight connected edges
  root.querySelectorAll(`.edge-path[data-src="${{nodeId}}"], .edge-path[data-dst="${{nodeId}}"]`).forEach(p => {{
    p.classList.add('highlighted');
    p.setAttribute('marker-end','url(#arrow-hl)');
  }});

  const nd = nodes.find(n => n.id === nodeId);
  if(nd) renderNodeDetail(nd);
  document.getElementById('right-panel').classList.remove('hidden');
}}

function renderNodeDetail(nd) {{
  const color = opColor(nd.category);
  let html = `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <div style="background:${{color}}33;border:1px solid ${{color}};border-radius:8px;padding:6px 14px;font-size:16px;font-weight:700;color:${{color}}">${{nd.op}}</div>
      ${{nd.flops > 0 ? `<div class="flops-badge">FLOPs: ${{fmtNum(nd.flops)}}</div>` : ''}}
    </div>
    <div class="info-row"><div class="info-key">节点名</div><div class="info-val">${{nd.name}}</div></div>
    <div class="info-row"><div class="info-key">算子类型</div><div class="info-val">${{nd.op}}</div></div>
    <div class="info-row"><div class="info-key">类别</div><div class="info-val">${{CAT_LABEL[nd.category]||nd.category}}</div></div>
  `;

  if(nd.input_shapes && nd.input_shapes.length) {{
    html += `<div class="section-title">输入张量</div>`;
    nd.input_shapes.forEach(t => {{
      html += `<div class="tensor-item"><div class="tensor-name">${{t.name}}</div><div class="tensor-shape">${{t.shape}}</div></div>`;
    }});
  }}
  if(nd.output_shapes && nd.output_shapes.length) {{
    html += `<div class="section-title">输出张量</div>`;
    nd.output_shapes.forEach(t => {{
      html += `<div class="tensor-item"><div class="tensor-name">${{t.name}}</div><div class="tensor-shape">${{t.shape}}</div></div>`;
    }});
  }}
  if(nd.attrs && Object.keys(nd.attrs).length) {{
    html += `<div class="section-title">属性</div>`;
    Object.entries(nd.attrs).forEach(([k,v]) => {{
      html += `<div class="attr-item"><div class="attr-key">${{k}}</div><div class="attr-val">${{JSON.stringify(v)}}</div></div>`;
    }});
  }}

  document.getElementById('node-detail').innerHTML = html;
}}

// ── Tooltip ──
const tooltip = document.getElementById('tooltip');
function showTooltip(ev, nd) {{
  const color = opColor(nd.category);
  tooltip.innerHTML = `<b style="color:${{color}}">${{nd.op}}</b><br>
    ${{nd.name}}<br>
    ${{nd.flops > 0 ? '⚡ FLOPs: ' + fmtNum(nd.flops) : ''}}`;
  tooltip.style.display = 'block';
  moveTooltip(ev);
}}
function moveTooltip(ev) {{
  tooltip.style.left = (ev.clientX + 14) + 'px';
  tooltip.style.top = (ev.clientY - 10) + 'px';
}}
function hideTooltip() {{ tooltip.style.display = 'none'; }}
svg.addEventListener('mousemove', moveTooltip);

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
document.getElementById('node-search').addEventListener('input', function() {{
  const q = this.value.toLowerCase().trim();
  root.querySelectorAll('.node-group').forEach(g => {{
    const nodeId = parseInt(g.dataset.id);
    const nd = nodes.find(n => n.id === nodeId);
    if(!nd) return;
    if(!q) {{ g.style.opacity = '1'; return; }}
    const match = nd.op.toLowerCase().includes(q) || nd.name.toLowerCase().includes(q);
    g.style.opacity = match ? '1' : '0.15';
  }});
}});

// ── Tab switching ──
function switchTab(tab) {{
  document.getElementById('tab-graph').classList.toggle('active', tab==='graph');
  document.getElementById('tab-stats').classList.toggle('active', tab==='stats');
  document.getElementById('graph-panel').style.display = tab==='graph' ? 'flex' : 'none';
  document.getElementById('right-panel').style.display = (tab==='graph' && !rightHidden) ? 'flex' : 'none';
  document.getElementById('stats-panel').classList.toggle('active', tab==='stats');
  if(tab==='graph') {{ setTimeout(fitView, 50); }}
  if(tab==='stats') buildStats();
}}

let rightHidden = false;
function togglePanel() {{
  rightHidden = !rightHidden;
  document.getElementById('right-panel').style.display = rightHidden ? 'none' : 'flex';
}}

// ── Stats ──
function buildStats() {{
  const maxCount = Math.max(...opStats.map(o => o.count), 1);
  const totalFlops = meta.total_flops;

  let html = `
  <div class="summary-cards">
    <div class="s-card">
      <div class="s-card-label">总节点数</div>
      <div class="s-card-value">${{meta.total_nodes}}</div>
      <div class="s-card-sub">算子节点</div>
    </div>
    <div class="s-card">
      <div class="s-card-label">参数量</div>
      <div class="s-card-value">${{fmtNum(meta.total_params)}}</div>
      <div class="s-card-sub">权重参数总数</div>
    </div>
    <div class="s-card">
      <div class="s-card-label">计算量 FLOPs</div>
      <div class="s-card-value">${{fmtNum(totalFlops)}}</div>
      <div class="s-card-sub">浮点运算次数（估算）</div>
    </div>
    <div class="s-card">
      <div class="s-card-label">算子种类</div>
      <div class="s-card-value">${{opStats.length}}</div>
      <div class="s-card-sub">不同算子类型数</div>
    </div>
  </div>`;

  // Model info
  html += `<div class="io-section"><div class="io-title">模型信息</div>`;
  const opsets = meta.opset.map(o => `${{o.domain}} v${{o.version}}`).join(', ');
  html += `
    <div class="io-item"><div class="io-name">IR版本</div><div class="io-meta">${{meta.ir_version}}</div></div>
    <div class="io-item"><div class="io-name">Opset</div><div class="io-meta">${{opsets}}</div></div>
    ${{meta.producer ? `<div class="io-item"><div class="io-name">生产工具</div><div class="io-meta">${{meta.producer}}</div></div>` : ''}}
    <div class="io-item"><div class="io-name">分析时间</div><div class="io-meta">${{meta.analyze_time}}</div></div>
  </div>`;

  // Inputs / Outputs
  html += `<div class="io-section"><div class="io-title">模型输入</div>`;
  meta.inputs.forEach(inp => {{
    html += `<div class="io-item"><div class="io-name">${{inp.name}}</div><div class="io-meta">形状: ${{inp.shape}} | 类型: ${{inp.dtype}}</div></div>`;
  }});
  html += `</div><div class="io-section"><div class="io-title">模型输出</div>`;
  meta.outputs.forEach(out => {{
    html += `<div class="io-item"><div class="io-name">${{out.name}}</div><div class="io-meta">形状: ${{out.shape}} | 类型: ${{out.dtype}}</div></div>`;
  }});
  html += `</div>`;

  // Op table
  html += `<div class="io-section"><div class="io-title">算子统计</div>
  <table class="op-table">
    <tr><th>算子</th><th>类别</th><th>数量</th><th class="bar-cell">占比</th></tr>`;
  opStats.forEach(op => {{
    const color = opColor(op.category);
    const pct = (op.count / maxCount * 100).toFixed(0);
    const pctTotal = (op.count / meta.total_nodes * 100).toFixed(1);
    html += `<tr>
      <td><span class="op-dot" style="background:${{color}}"></span>${{op.op}}</td>
      <td style="color:var(--text2)">${{CAT_LABEL[op.category]||op.category}}</td>
      <td style="font-weight:700">${{op.count}} <span style="color:var(--text2);font-size:11px">(${{pctTotal}}%)</span></td>
      <td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div></td>
    </tr>`;
  }});
  html += `</table></div>`;

  document.getElementById('stats-content').innerHTML = html;
}}

// ── Number formatting ──
function fmtNum(n) {{
  if(n >= 1e12) return (n/1e12).toFixed(2) + ' T';
  if(n >= 1e9)  return (n/1e9).toFixed(2)  + ' G';
  if(n >= 1e6)  return (n/1e6).toFixed(2)  + ' M';
  if(n >= 1e3)  return (n/1e3).toFixed(2)  + ' K';
  return String(n);
}}

// ── Init ──
buildGraph();
buildLegend();
setTimeout(fitView, 100);
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML报告已生成: {output_path}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ONNX模型分析器 - 生成交互式HTML可视化报告")
    parser.add_argument("model", help="ONNX模型文件路径")
    parser.add_argument("--output", "-o", default=None,
                        help="输出HTML文件路径（默认与模型同名.html）")
    parser.add_argument("--print-stats", action="store_true",
                        help="同时在终端输出统计信息")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"❌ 错误：找不到文件 {args.model}")
        sys.exit(1)

    if not args.model.lower().endswith(".onnx"):
        print("⚠️  警告：文件不以 .onnx 结尾，继续尝试解析...")

    output_path = args.output
    if output_path is None:
        base = os.path.splitext(args.model)[0]
        output_path = base + "_analysis.html"

    print(f"🔍 正在解析: {args.model}")
    meta, nodes_data, edges, op_stats = analyze_onnx(args.model)

    print(f"📊 节点数: {meta['total_nodes']}")
    print(f"📦 参数量: {fmt_num(meta['total_params'])}")
    print(f"⚡ FLOPs:  {fmt_num(meta['total_flops'])} (估算)")

    if args.print_stats:
        print("\n── 算子统计 ──")
        for op in op_stats:
            print(f"  {op['op']:30s} x{op['count']}")

    generate_html(meta, nodes_data, edges, op_stats, output_path)
    print(f"\n✨ 用浏览器打开: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
