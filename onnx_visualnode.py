#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onnx_visualnode.py - ONNX Debugger Single File Version
合并了所有模块功能的单文件版本，用于捕获和可视化ONNX模型推理过程中所有节点的中间张量值。

Usage:
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
# 模块1: graph_patcher - 图修改工具
# ============================================================================

def patch_model_expose_all_intermediates(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    将所有中间张量(value_info)注册为图输出。
    这是捕获所有节点激活值的核心技巧。
    """
    # 形状推断填充可能缺失的value_info条目
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
# 模块2: model_loader - 模型加载器
# ============================================================================

def load_model(model_path: str) -> onnx.ModelProto:
    """加载ONNX模型并尝试形状推断。"""
    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[model_loader] shape inference warning: {e}")
    return model


# ============================================================================
# 模块3: runner - 推理执行器
# ============================================================================

class OnnxRunner:
    def __init__(self, model_path: str):
        model = onnx.load(model_path)
        patched_model = patch_model_expose_all_intermediates(model)

        # 禁用图优化以防止节点融合
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        # 直接从内存加载修改后的模型 — 无需临时文件
        self.session = ort.InferenceSession(
            patched_model.SerializeToString(),
            sess_options=sess_options,
        )
        self.model = model

    def run_from_npy(self, npy_path: str) -> dict:
        """加载input.npy并返回所有张量(输入 + 每个中间层)。"""
        input_data = np.load(npy_path, allow_pickle=True)

        # 支持dict-in-npy(多输入)或普通数组(单输入)
        if input_data.dtype == object:
            inputs = input_data.item()          # {name: array}
        else:
            input_name = self.session.get_inputs()[0].name
            inputs = {input_name: input_data}

        output_names = [o.name for o in self.session.get_outputs()]
        results = self.session.run(output_names, inputs)

        # 合并原始输入以便每个节点的输入张量都可访问
        all_tensors = {**inputs, **dict(zip(output_names, results))}
        return all_tensors


# ============================================================================
# 模块4: node_info - 节点信息提取
# ============================================================================

def get_node_static_info(node: onnx.NodeProto, idx: int) -> dict:
    """返回 {node_id, op_type, attrs, input_names, output_names}。"""
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
# 模块5: tensor_viewer - 张量统计
# ============================================================================

def tensor_stats(arr: np.ndarray) -> dict:
    """返回数值张量的min/max/mean/std/abs_mean。"""
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
    返回单个张量的详细描述字典:
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
# 模块6: debugger - 主调试器接口
# ============================================================================

class OnnxDebugger:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.runner = OnnxRunner(model_path)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------
    def run(self, npy_path: str) -> dict:
        """
        运行推理并返回每个节点的结构化字典，包含实际张量
        值(形状 + 统计信息)用于每个输入和输出。
        """
        all_tensors = self.runner.run_from_npy(npy_path)   # {name: np.ndarray}

        result = {}
        for idx, node in enumerate(self.model.graph.node):
            info = get_node_static_info(node, idx)
            node_id = info["node_id"]

            inputs_data = {}
            for name in info["input_names"]:
                if name:  # 跳过空的可选输入
                    arr = all_tensors.get(name)
                    desc = describe_tensor(name, arr)
                    if desc.get("available"):
                        # 仅保留可序列化的键
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
    # 便捷辅助方法
    # ------------------------------------------------------------------
    def inspect_node(self, node_id: str, npy_path: str):
        """漂亮打印单个节点的I/O张量。"""
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
# 模块7: html_builder - HTML报告生成
# ============================================================================

def _safe_json(obj):
    """序列化为JSON，优雅地转换不可序列化的类型。"""
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


def build_html(
    debug_result: dict,
    model_path: str,
    npy_path: str,
    output_path: str,
):
    """
    debug_result  – OnnxDebugger.run()返回的字典
    model_path    – .onnx路径(用于显示)
    npy_path      – 输入.npy路径(用于显示)
    output_path   – .html文件写入位置
    """

    # 为每个节点字典添加id和category以便JS使用
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

    data_json = _safe_json({
        "model":    os.path.basename(model_path),
        "npy":      os.path.basename(npy_path),
        "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nodes":    nodes_list,
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
<title>ONNX Debugger – {model_name}</title>
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
    <div class="tab" id="tab-stats" onclick="switchTab('stats')">Statistics</div>
  </div>
</header>

<div class="main">
  <!-- left: node list -->
  <div id="list-panel">
    <div id="search-wrap">
      <input id="node-search" type="text" placeholder="🔍 Filter nodes…">
    </div>
    <div id="node-list"></div>
  </div>

  <!-- right: detail + stats -->
  <div style="flex:1;display:flex;flex-direction:column;overflow:hidden">
    <div id="detail-panel">
      <div class="no-sel">← Select a node to inspect its tensors</div>
    </div>
    <div id="stats-panel"></div>
  </div>
</div>

<script>
const RAW = {data_json};
const nodes = RAW.nodes;

// ── colour map ──────────────────────────────────────────────────────────────
const CAT_COLOR = {{
  conv:"#3b82f6",gemm:"#8b5cf6",act:"#22c55e",pool:"#06b6d4",
  norm:"#f59e0b",eltwise:"#ec4899",shape:"#a78bfa",upsample:"#10b981",
  rnn:"#f97316",reduce:"#84cc16",attention:"#e11d48",other:"#6b7280"
}};
const CAT_LABEL = {{
  conv:"Conv",gemm:"FC/MatMul",act:"Activation",pool:"Pooling",
  norm:"Normalization",eltwise:"Element-wise",shape:"Shape Ops",
  upsample:"Upsample",rnn:"RNN",reduce:"Reduce",attention:"Attention",other:"Other"
}};
function catColor(c){{ return CAT_COLOR[c]||"#6b7280"; }}

// ── header info ─────────────────────────────────────────────────────────────
document.getElementById("h-model").textContent = RAW.model + "  ·  " + RAW.npy;
document.getElementById("h-nodes").textContent = nodes.length;
document.getElementById("h-time").textContent  = RAW.time;

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

// ── statistics tab ───────────────────────────────────────────────────────────
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

// ── tab switch ───────────────────────────────────────────────────────────────
function switchTab(t){{
  document.getElementById("tab-debug").classList.toggle("active", t==="debug");
  document.getElementById("tab-stats").classList.toggle("active", t==="stats");
  document.getElementById("list-panel").style.display = t==="debug"?"flex":"none";
  document.getElementById("detail-panel").style.display = t==="debug"?"block":"none";
  document.getElementById("stats-panel").classList.toggle("active", t==="stats");
  if(t==="stats") buildStats();
}}

// ── init ─────────────────────────────────────────────────────────────────────
buildList();
if(nodes.length>0) selectNode(0);
</script>
</body>
</html>"""


# ============================================================================
# 模块8: CLI - 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ONNX Debugger – capture every node's actual tensor values and produce an HTML report."
    )
    parser.add_argument("model",  help="Path to the ONNX model (.onnx)")
    parser.add_argument("input",  help="Path to the input data (.npy)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML path (default: <model>_debug.html)")
    parser.add_argument("--inspect", "-i", default=None, metavar="NODE_ID",
                        help="Also pretty-print details for a single node")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Dump the raw debug result to <model>_debug.json")
    args = parser.parse_args()

    # ── 验证输入 ──────────────────────────────────────────────────────
    for path, label in [(args.model, "model"), (args.input, "input npy")]:
        if not os.path.isfile(path):
            print(f"❌ {label} not found: {path}")
            sys.exit(1)

    # ── 默认输出路径 ──────────────────────────────────────────────────
    out_html = args.output
    if out_html is None:
        base = os.path.splitext(args.model)[0]
        out_html = base + "_debug.html"

    # ── 运行调试器 ─────────────────────────────────────────────────────────
    print(f"[*] Loading model  : {args.model}")
    print(f"[*] Input data     : {args.input}")

    debugger = OnnxDebugger(args.model)

    print("[*] Running inference (all intermediate outputs enabled)...")
    result = debugger.run(args.input)

    n_nodes   = len(result)
    n_tensors = sum(len(v["inputs"]) + len(v["outputs"]) for v in result.values())
    print(f"[OK] Captured {n_nodes} nodes, {n_tensors} tensor snapshots")

    # ── 可选: 检查单个节点 ────────────────────────────────────────
    if args.inspect:
        debugger.inspect_node(args.inspect, args.input)

    # ── 可选: 导出JSON ──────────────────────────────────────────────────
    if args.json:
        json_path = os.path.splitext(args.model)[0] + "_debug.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[*] JSON dump      : {json_path}")

    # ── 生成HTML报告 ─────────────────────────────────────────────────
    print("[*] Generating HTML report...")
    build_html(result, args.model, args.input, out_html)
    print(f"\n[OK] Open in browser: {os.path.abspath(out_html)}")


if __name__ == "__main__":
    main()
