#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py  –  ONNX Debugger command-line entry point.

Usage:
    python cli.py resnet18.onnx input.npy
    python cli.py resnet18.onnx input.npy --output debug_report.html
    python cli.py resnet18.onnx input.npy --inspect Conv_0
"""

import sys
import os
import argparse
import json

# Make sure the package root is importable when cli.py lives beside onnx_debugger/
sys.path.insert(0, os.path.dirname(__file__))

from onnx_debugger.debugger import OnnxDebugger
from onnx_debugger.report.html_builder import build_html


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

    # ── validate inputs ──────────────────────────────────────────────────────
    for path, label in [(args.model, "model"), (args.input, "input npy")]:
        if not os.path.isfile(path):
            print(f"❌ {label} not found: {path}")
            sys.exit(1)

    # ── default output path ──────────────────────────────────────────────────
    out_html = args.output
    if out_html is None:
        base = os.path.splitext(args.model)[0]
        out_html = base + "_debug.html"

    # ── run debugger ─────────────────────────────────────────────────────────
    print(f"[*] Loading model  : {args.model}")
    print(f"[*] Input data     : {args.input}")

    debugger = OnnxDebugger(args.model)

    print("[*] Running inference (all intermediate outputs enabled)...")
    result = debugger.run(args.input)

    n_nodes   = len(result)
    n_tensors = sum(len(v["inputs"]) + len(v["outputs"]) for v in result.values())
    print(f"[OK] Captured {n_nodes} nodes, {n_tensors} tensor snapshots")

    # ── optional: inspect single node ────────────────────────────────────────
    if args.inspect:
        debugger.inspect_node(args.inspect, args.input)

    # ── optional: dump JSON ──────────────────────────────────────────────────
    if args.json:
        json_path = os.path.splitext(args.model)[0] + "_debug.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[*] JSON dump      : {json_path}")

    # ── generate HTML report ─────────────────────────────────────────────────
    print("[*] Generating HTML report...")
    build_html(result, args.model, args.input, out_html)
    print(f"\n[OK] Open in browser: {os.path.abspath(out_html)}")


if __name__ == "__main__":
    main()
