#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspector/tensor_viewer.py
Compute shape / dtype / statistics for a numpy tensor.
"""

import numpy as np


def tensor_stats(arr: np.ndarray) -> dict:
    """Return min/max/mean/std/abs_mean for a numeric tensor."""
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
    Return a rich description dict for a single tensor:
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
