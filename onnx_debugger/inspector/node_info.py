#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspector/node_info.py
Extract static per-node metadata: op_type, attributes, input/output names.
"""

import onnx


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
