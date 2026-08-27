# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Engine-format bridge: import/export of the neural morph network exchange
format (``*.nmn``, global mode) and import of plain ONNX MLPs.

Import mapping (the .nmn file stores counts, not names):
    bones  -> the first N entries of Inputs > Bones (in list order)
    curves -> the first N enabled Curve Inputs shape keys (in list order)
    morphs -> the first N enabled Morph Target shape keys (in list order)

Export writes the network plus a small JSON sidecar with the name mapping so
round-trips (and re-imports) stay consistent.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import bridge
from .core.features import BoneFeature, FeatureSpec
from .core.network import NeuralMorphRegressor
from .core.ue_nmn import (
    NmnModel, NmnError, build_nmn, parse_nmn, build_ue_input, run_nmn,
    linear_sequence, MODE_GLOBAL,
)
from .core.onnx_io import parse_onnx, run_onnx, OnnxError


class NmnPredictor:
    """Wraps a parsed .nmn model behind the shared predictor interface."""

    def __init__(self, nmn_model, morph_deltas, clamp=True):
        self.nmn = nmn_model
        self.morph_deltas = morph_deltas
        self.clamp = clamp

    def predict(self, x):
        weights = run_nmn(self.nmn, x, clamp=self.clamp)  # (M,) or (M, N)
        if weights.ndim == 1:
            weights = weights[:, None]
        return self.morph_deltas @ weights


class OnnxPredictor:
    """Wraps a parsed ONNX graph behind the shared predictor interface."""

    def __init__(self, graph, morph_deltas, clamp=True):
        self.graph = graph
        self.morph_deltas = morph_deltas
        self.clamp = clamp

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        weights = np.stack([run_onnx(self.graph, x[:, i], clamp_outputs=None)
                            for i in range(x.shape[1])], axis=1)
        if self.clamp:
            weights = np.clip(weights, 0.0, 1.0)
        return self.morph_deltas @ weights


def _mapping_spec(settings, num_bones, num_curves, num_morphs):
    """Build a FeatureSpec for the imported network from the current lists."""
    bones = [e for e in settings.bones][:num_bones]
    curves = [e.name for e in settings.curve_inputs if e.use][:num_curves]
    morphs = [e.name for e in settings.morph_targets if e.use][:num_morphs]
    if len(bones) < num_bones:
        raise ValueError(
            "network uses %d bones but only %d bone entries exist; sync "
            "bones in Inputs first" % (num_bones, len(bones)))
    if len(curves) < num_curves:
        raise ValueError(
            "network uses %d curves but only %d curve inputs are enabled"
            % (num_curves, len(curves)))
    if len(morphs) < num_morphs:
        raise ValueError(
            "network outputs %d morph weights but only %d morph targets are "
            "enabled" % (num_morphs, len(morphs)))
    spec = FeatureSpec()
    spec.bones = [BoneFeature(b.name, use_rotation=True,
                              use_translation=b.use_translation,
                              use_scale=b.use_scale) for b in bones]
    spec.curve_names = curves
    return spec, morphs


def _ue_input_builder(local_mode=False):
    def build(rotations, translations, scales, curves):
        return build_ue_input(rotations, curves,
                              num_curves=len(curves),
                              local_mode=local_mode)
    return build


def import_nmn(settings, path):
    with open(path, "rb") as f:
        data = f.read()
    model = parse_nmn(data)
    if model.is_local_mode:
        print("[Blender ML Deformer] warning: local-mode networks are imported but "
              "can only run their main network; groups are ignored")
    spec, morph_names = _mapping_spec(settings, model.num_bones,
                                      model.num_curves, model.num_morphs)
    morph_deltas = bridge.compute_morph_deltas(settings.mesh, morph_names)
    predictor = NmnPredictor(model, morph_deltas,
                             clamp=settings.engine_clamp_weights)
    bridge.ACTIVE_MODEL = predictor
    bridge.ACTIVE_SPEC = spec
    bridge.ACTIVE_INPUT = _ue_input_builder(model.is_local_mode)
    bridge.ACTIVE_MORPH_DELTAS = morph_deltas
    bridge.ACTIVE_KIND = "ue"
    bridge._MORPH_NAMES = morph_names
    settings.model_kind = "NEURAL"
    settings.is_trained = True
    settings.num_features = model.num_main_inputs()
    settings.num_vertices = morph_deltas.shape[0] // 3
    settings.training_loss = 0.0
    settings.max_vertex_error = 0.0
    return os.path.basename(path)


def import_onnx(settings, path):
    with open(path, "rb") as f:
        data = f.read()
    graph = parse_onnx(data)
    num_morphs = len([e.name for e in settings.morph_targets if e.use])
    if num_morphs == 0:
        raise ValueError("enable at least one Morph Target shape key first")
    # Infer the expected feature count from the first MatMul/Gemm weight.
    expected = None
    for node in graph["nodes"]:
        if node["op"] in ("Gemm", "MatMul") and node["inputs"]:
            w = graph["initializers"].get(node["inputs"][1])
            if w is not None and w.ndim == 2:
                expected = w.shape[0] if node["op"] == "Gemm" else w.shape[0]
                break
    spec = bridge.build_spec(settings)
    num_curves = len(spec.curve_names)
    num_bones = len(spec.bones)
    if expected is not None and expected != num_bones * 6 + num_curves:
        raise ValueError(
            "ONNX expects %d inputs but the current setup provides %d "
            "(bones %d x 6 + curves %d); adjust the Inputs section"
            % (expected, num_bones * 6 + num_curves, num_bones, num_curves))
    morph_names = [e.name for e in settings.morph_targets if e.use]
    morph_deltas = bridge.compute_morph_deltas(settings.mesh, morph_names)
    predictor = OnnxPredictor(graph, morph_deltas,
                              clamp=settings.engine_clamp_weights)
    bridge.ACTIVE_MODEL = predictor
    bridge.ACTIVE_SPEC = spec
    bridge.ACTIVE_INPUT = _ue_input_builder(False)
    bridge.ACTIVE_MORPH_DELTAS = morph_deltas
    bridge.ACTIVE_KIND = "ue"
    bridge._MORPH_NAMES = morph_names
    settings.model_kind = "NEURAL"
    settings.is_trained = True
    settings.num_features = num_bones * 6 + num_curves
    settings.num_vertices = morph_deltas.shape[0] // 3
    settings.training_loss = 0.0
    settings.max_vertex_error = 0.0
    return os.path.basename(path)


def export_nmn_iter(settings, path):
    """Export the trained neural model as a global-mode .nmn file.

    The network must be retrained on the engine input layout (6 floats per
    bone), so this re-fits an ELU MLP on the same training data and then
    serializes it.  Yields progress 0..1."""
    if bridge.ACTIVE_KIND != "neural" or not isinstance(
            bridge.ACTIVE_MODEL, NeuralMorphRegressor):
        raise ValueError("export requires a Neural model trained in Blender ML Deformer")
    cache = bridge.TRAINING_CACHE
    if cache is None:
        raise ValueError("the training cache is gone; regenerate training data "
                         "and retrain before exporting")
    spec = cache["spec"]
    morph_names = cache["morph_names"]
    raw_rotations = cache["raw_rotations"]      # (F, B, 3) axis-angle
    raw_curves = cache["raw_curves"]            # (F, C)
    num_curves = len(spec.curve_names)
    yield 0.02

    X_ue = np.stack([build_ue_input(raw_rotations[i], raw_curves[i],
                                    num_curves=num_curves, local_mode=False)
                     for i in range(raw_rotations.shape[0])], axis=1)
    hidden = bridge.ACTIVE_MODEL.hidden_sizes
    morph_deltas = bridge.ACTIVE_MODEL.morph_deltas
    export_model = NeuralMorphRegressor(X_ue.shape[0], morph_deltas, hidden,
                                        activation="elu")
    fit = export_model.fit_iter(X_ue, cache["Y"],
                                iterations=settings.iterations,
                                batch_size=settings.batch_size,
                                learning_rate=settings.learning_rate,
                                regularization=settings.regularization,
                                clamp=True,
                                chunk=max(1, settings.iterations // 100),
                                seed=settings.random_seed or 0)
    for frac, _loss in fit:
        yield 0.05 + frac * 0.75

    nmn = NmnModel()
    nmn.mode = MODE_GLOBAL
    nmn.num_morphs = morph_deltas.shape[1]
    nmn.num_morphs_per_bone = 0
    nmn.num_bones = len(spec.bones)
    nmn.num_curves = num_curves
    nmn.num_groups = 0
    nmn.num_items_per_group = 0
    nmn.num_floats_per_curve = 1
    nmn.input_mean = export_model.input_mean
    nmn.input_std = export_model.input_std
    nmn.main_layers = linear_sequence(hidden, export_model.net.weights,
                                      export_model.net.biases, activation="elu")
    with open(path, "wb") as f:
        f.write(build_nmn(nmn))

    sidecar = {
        "bone_names": spec.bone_names,
        "curve_names": list(spec.curve_names),
        "morph_names": list(morph_names),
        "input_mean": nmn.input_mean.tolist(),
        "input_std": nmn.input_std.tolist(),
    }
    side_path = os.path.splitext(path)[0] + ".bmd_ue.json"
    with open(side_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    yield 1.0
