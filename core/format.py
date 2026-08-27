# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Blender ML Deformer's own model format: pose_model.json + pose_model.npz.

The JSON carries the feature spec, model kind, morph names and stats; the
npz carries the numeric arrays (numpy ships with Blender; JSON would be far
too slow for per-vertex matrices).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .features import FeatureSpec

INFO_NAME = "pose_model.json"
WEIGHTS_NAME = "pose_model.npz"


def save_model(directory, name, model_kind, spec, arrays, morph_names, stats):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    info = {
        "name": name,
        "model_kind": model_kind,  # "linear" | "neural"
        "feature_spec": spec.to_dict(),
        "morph_names": list(morph_names),
        "stats": {k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in stats.items()},
        "version": 1,
    }
    with open(os.path.join(directory, INFO_NAME), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    np.savez_compressed(os.path.join(directory, WEIGHTS_NAME), **arrays)
    return os.path.join(directory, INFO_NAME)


def load_model(directory):
    directory = os.path.abspath(directory)
    info_path = os.path.join(directory, INFO_NAME)
    weights_path = os.path.join(directory, WEIGHTS_NAME)
    if not os.path.isfile(info_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError(
            "No Blender ML Deformer model in %r (need %s + %s)"
            % (directory, INFO_NAME, WEIGHTS_NAME))
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    arrays = {k: np.asarray(v) for k, v in np.load(weights_path).items()}
    return {
        "name": info.get("name", ""),
        "model_kind": info.get("model_kind"),
        "spec": FeatureSpec.from_dict(info.get("feature_spec", {})),
        "arrays": arrays,
        "morph_names": info.get("morph_names", []),
        "stats": info.get("stats", {}),
    }
