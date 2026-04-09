#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report/html_builder.py
Build a self-contained HTML debug report from the OnnxDebugger result dict.
"""

import json
import os
from datetime import datetime


# ─── helpers ──────────────────────────────────────────────────────────────────

def _safe_json(obj):
    """Serialise to JSON, converting non-serialisable types gracefully."""
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


# ─── public entry point ────────────────────────────────────────────────────────

def build_html(
    debug_result: dict,
    model_path: str,
    npy_path: str,
    output_path: str,
):
    """
    debug_result  – the dict returned by OnnxDebugger.run()
    model_path    – path to .onnx (for display)
    npy_path      – path to input .npy (for display)
    output_path   – where to write the .html file
    """

    # Annotate each node dict with its id and category so JS can use it
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


# ─── HTML template ─────────────────────────────────────────────────────────────

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
