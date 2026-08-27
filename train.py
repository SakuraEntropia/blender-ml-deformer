# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Training orchestration: data generation, model fitting, baking, and the
own-format save/load. Long operations are generators yielding progress so
the UI can run them inside modal operators."""

from __future__ import annotations

import json
import os

import bpy
import numpy as np

from . import bridge
from .core.features import FeatureSpec
from .core.regressor import LinearRegressor
from .core.network import NeuralMorphRegressor
from .core import format as fmt


def _sample_morph_weights(rng, n, zero_prob):
    w = rng.random(n)
    mask = rng.random(n) < zero_prob
    w[mask] = 0.0
    if n > 0 and not w.any():
        w[int(rng.integers(0, n))] = 1.0
    return w


def generate_training_data_iter(settings):
    """Sample poses, record deltas. Stores bridge.TRAINING_CACHE and updates
    stats. Yields progress 0..1."""
    armature = settings.armature
    mesh = settings.mesh
    if armature is None or mesh is None:
        raise ValueError("Pick an Armature and a Mesh in the Setup section first")
    spec = bridge.build_spec(settings)
    curve_names = spec.curve_names
    morph_names = [e.name for e in settings.morph_targets if e.use]
    neural = settings.model_kind == "NEURAL"
    if neural and not morph_names:
        raise ValueError("Neural model needs morph targets: enable shape keys "
                         "in Inputs > Morph Targets (or switch to Linear)")

    depsgraph = bridge.get_depsgraph()
    base = bridge.capture_base(settings, spec, depsgraph)
    V = base.shape[0]
    if V == 0:
        raise ValueError("The mesh has no vertices")

    n_clips = sum(c.num_frames for c in settings.training_clips if c.use) \
        if settings.use_clip_sampling else 0
    F = settings.num_random_poses + n_clips + 1  # +1: reference pose
    if F <= 1:
        raise ValueError("No training poses configured")

    rng = np.random.default_rng(settings.random_seed if settings.random_seed else None)
    X = np.empty((spec.num_features, F))
    D = np.empty((V * 3, F))
    Y = np.empty((len(morph_names), F)) if (neural and morph_names) else None
    raw_rotations = np.zeros((F, len(spec.bones), 3))
    raw_curves = np.zeros((F, len(curve_names)))

    pose_snap = bridge.snapshot_pose(armature)
    key_snap = bridge.snapshot_keys(mesh)
    frame = 0
    try:
        # reference pose: zero input -> zero deltas
        X[:, frame] = spec.build_vector(np.zeros((len(spec.bones), 3)),
                                        np.zeros((len(spec.bones), 3)),
                                        np.zeros((len(spec.bones), 3)),
                                        np.zeros(len(curve_names)))
        D[:, frame] = 0.0
        if Y is not None:
            Y[:, frame] = 0.0
        frame += 1

        def _record():
            nonlocal frame
            rotations, translations, scales = spec.sample_pose(
                rng, bridge.rotation_ranges(settings))
            bridge.write_pose(armature, spec, rotations, translations, scales)
            curve_vals = rng.random(len(curve_names)) if curve_names else None
            bridge.set_key_values(mesh, curve_names, curve_vals)
            if neural:
                w = _sample_morph_weights(rng, len(morph_names),
                                          settings.morph_zero_prob)
                bridge.set_key_values(mesh, morph_names, w)
                Y[:, frame] = w
            depsgraph.update()
            pos = bridge.eval_positions(mesh, depsgraph)
            if pos.shape[0] != V:
                raise ValueError(
                    "Mesh vertex count changed during evaluation (%d -> %d). "
                    "Topology-changing modifiers are not supported."
                    % (V, pos.shape[0]))
            D[:, frame] = (pos - base).ravel()
            X[:, frame] = spec.build_vector(
                rotations, translations, scales,
                curve_vals if curve_vals is not None else np.zeros(0))
            raw_rotations[frame] = rotations
            raw_curves[frame] = curve_vals if curve_vals is not None else np.zeros(0)
            frame += 1

        for _ in range(settings.num_random_poses):
            _record()
            yield frame / F * 0.9

        if settings.use_clip_sampling:
            scene = bpy.context.scene
            prev_action = armature.animation_data.action if armature.animation_data else None
            prev_frame = scene.frame_current
            try:
                for clip in settings.training_clips:
                    if not clip.use or frame >= F:
                        continue
                    act = bpy.data.actions.get(clip.name)
                    if act is None:
                        continue
                    if armature.animation_data is None:
                        armature.animation_data_create()
                    armature.animation_data.action = act
                    denom = max(1, clip.num_frames - 1)
                    for k in range(clip.num_frames):
                        if frame >= F:
                            break
                        fr = round(clip.frame_start
                                   + (clip.frame_end - clip.frame_start) * k / denom)
                        scene.frame_set(fr)
                        depsgraph.update()
                        rotations, translations, scales = bridge.read_pose(armature, spec)
                        curve_vals = rng.random(len(curve_names)) if curve_names else None
                        bridge.set_key_values(mesh, curve_names, curve_vals)
                        if neural:
                            w = _sample_morph_weights(rng, len(morph_names),
                                                      settings.morph_zero_prob)
                            bridge.set_key_values(mesh, morph_names, w)
                            Y[:, frame] = w
                        depsgraph.update()
                        pos = bridge.eval_positions(mesh, depsgraph)
                        if pos.shape[0] != V:
                            raise ValueError(
                                "Mesh vertex count changed during evaluation "
                                "(%d -> %d). Topology-changing modifiers are "
                                "not supported." % (V, pos.shape[0]))
                        D[:, frame] = (pos - base).ravel()
                        X[:, frame] = spec.build_vector(
                            rotations, translations, scales,
                            curve_vals if curve_vals is not None else np.zeros(0))
                        raw_rotations[frame] = rotations
                        raw_curves[frame] = curve_vals if curve_vals is not None else np.zeros(0)
                        frame += 1
                        yield frame / F * 0.9
            finally:
                if armature.animation_data is not None:
                    armature.animation_data.action = prev_action
                scene.frame_set(prev_frame)
                depsgraph.update()

        bridge.TRAINING_CACHE = {
            "X": X[:, :frame],
            "D": D[:, :frame],
            "Y": Y[:, :frame] if Y is not None else None,
            "raw_rotations": raw_rotations[:frame],
            "raw_curves": raw_curves[:frame],
            "base": base,
            "spec": spec,
            "morph_names": morph_names,
            "curve_names": curve_names,
        }
        settings.num_training_frames = frame
        settings.num_vertices = V
        settings.num_features = spec.num_features
        yield 1.0
    finally:
        bridge.restore_pose(armature, pose_snap)
        bridge.restore_keys(mesh, key_snap)
        depsgraph.update()


def train_model_iter(settings):
    """Fit the selected model on the cache. Yields progress 0..1."""
    cache = bridge.TRAINING_CACHE
    if cache is None:
        raise ValueError("Generate training data first (Training section)")
    spec = cache["spec"]
    X = cache["X"]
    D = cache["D"]
    morph_names = cache["morph_names"]
    yield 0.05

    if settings.model_kind == "LINEAR":
        model = LinearRegressor()
        model.fit(X, D, regularization=settings.regularization)
        bridge.ACTIVE_MORPH_DELTAS = None
        bridge.ACTIVE_KIND = "linear"
        yield 0.8
    else:
        if not morph_names:
            raise ValueError("Neural model needs morph targets; enable shape "
                             "keys in Inputs > Morph Targets and regenerate data")
        morph_deltas = bridge.compute_morph_deltas(settings.mesh, morph_names)
        if morph_deltas.shape[1] == 0:
            raise ValueError("No morph target shape keys found on the mesh")
        hidden = bridge.parse_hidden_layers(settings.hidden_layers)
        model = NeuralMorphRegressor(spec.num_features, morph_deltas, hidden,
                                     activation="relu")
        fit = model.fit_iter(X, cache["Y"],
                             iterations=settings.iterations,
                             batch_size=settings.batch_size,
                             learning_rate=settings.learning_rate,
                             regularization=settings.regularization,
                             clamp=settings.clamp_morph_weights,
                             chunk=max(1, settings.iterations // 200))
        for frac, _loss in fit:
            yield 0.1 + frac * 0.7
        bridge.ACTIVE_MORPH_DELTAS = morph_deltas
        bridge.ACTIVE_KIND = "neural"

    if settings.model_kind == "NEURAL":
        w_pred = model.predict_weights(X)
        w_true = np.clip(cache["Y"], 0.0, 1.0)
        loss = float(np.mean((w_pred - w_true) ** 2))
        d_true = model.morph_deltas @ w_true
        max_err = float(np.max(np.abs(model.predict(X) - d_true)))
    else:
        pred = model.predict(X)
        loss = float(np.mean((pred - D) ** 2))
        max_err = float(np.max(np.abs(pred - D)))

    bridge.ACTIVE_MODEL = model
    bridge.ACTIVE_SPEC = spec
    bridge.ACTIVE_INPUT = (lambda rot, trans, scales, curves:
                           spec.build_vector(rot, trans, scales, curves))
    bridge._MORPH_NAMES = list(morph_names)
    settings.is_trained = True
    settings.training_loss = loss
    settings.max_vertex_error = max_err
    yield 1.0


def bake_shape_keys_iter(settings):
    """Sample poses, run the active model, bake deltas as relative shape keys
    on a new object, and write a pose library JSON."""
    if bridge.ACTIVE_MODEL is None or bridge.ACTIVE_INPUT is None:
        raise ValueError("Train or import a model first")
    mesh = settings.mesh
    armature = settings.armature
    spec = bridge.ACTIVE_SPEC
    if mesh is None or armature is None or spec is None:
        raise ValueError("Pick an Armature and a Mesh in the Setup section first")
    depsgraph = bridge.get_depsgraph()
    base = bridge.capture_base(settings, spec, depsgraph)
    src_eval = mesh.evaluated_get(depsgraph)
    me = src_eval.to_mesh(depsgraph=depsgraph)
    try:
        baked_me = bpy.data.meshes.new_from_object(src_eval)
    finally:
        src_eval.to_mesh_clear()
    if baked_me is None or len(baked_me.vertices) != len(base):
        baked_me = bpy.data.meshes.new(mesh.name + "_BMDBaked")
        baked_me.vertices.add(len(base))
    if baked_me.shape_keys is not None:
        bpy.data.shape_keys.remove(baked_me.shape_keys, do_unlink=True)
    baked_me.name = mesh.name + "_BMDBaked"
    baked_me.vertices.foreach_set("co", base.ravel())
    baked_obj = bpy.data.objects.new(baked_me.name, baked_me)
    bpy.context.scene.collection.objects.link(baked_obj)
    baked_obj.shape_key_add(name="Basis", from_mix=False)
    baked_me.update()

    rng = np.random.default_rng(settings.random_seed if settings.random_seed else None)
    pose_snap = bridge.snapshot_pose(armature)
    key_snap = bridge.snapshot_keys(mesh)
    library = {"spec": spec.to_dict(), "poses": []}
    K = settings.num_bake_poses
    try:
        for i in range(K):
            rotations, translations, scales = spec.sample_pose(
                rng, bridge.rotation_ranges(settings))
            bridge.write_pose(armature, spec, rotations, translations, scales)
            curve_vals = rng.random(len(spec.curve_names)) if spec.curve_names else None
            bridge.set_key_values(mesh, spec.curve_names, curve_vals)
            depsgraph.update()
            x = bridge.ACTIVE_INPUT(rotations, translations, scales,
                                    curve_vals if curve_vals is not None else np.zeros(0))
            delta = bridge.model_predict(x[:, None])[:, 0]
            name = "%s%03d" % (bridge.BAKE_KEY_PREFIX, i)
            kb = baked_obj.shape_key_add(name=name, from_mix=False)
            kb.data.foreach_set("co", delta)
            library["poses"].append({"name": name, "input": x.tolist()})
            yield (i + 1) / K
        baked_me.update()
        out_dir = settings.model_dir or (os.path.dirname(bpy.data.filepath)
                                         if bpy.data.filepath else "")
        if out_dir:
            path = os.path.join(out_dir, "BMD_pose_library.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(library, f, indent=2)
        else:
            print("[Blender ML Deformer] blend not saved and no Model Directory set; "
                  "pose library not written")
    finally:
        bridge.restore_pose(armature, pose_snap)
        bridge.restore_keys(mesh, key_snap)
        depsgraph.update()


# ---------------------------------------------------------------------------
# Own format save/load
# ---------------------------------------------------------------------------

def export_model(settings):
    if bridge.ACTIVE_MODEL is None or bridge.ACTIVE_SPEC is None:
        raise ValueError("Train or import a model first")
    directory = settings.model_dir
    if not directory:
        raise ValueError("Set a Model Directory in Preview & IO first")
    if bridge.ACTIVE_KIND == "linear":
        arrays = {"matrix": np.asarray(bridge.ACTIVE_MODEL.matrix)}
    elif bridge.ACTIVE_KIND == "neural":
        arrays = bridge.ACTIVE_MODEL.param_dict()
    else:
        raise ValueError("engine-imported networks must be re-exported via the "
                         "Engine Bridge section")
    stats = {
        "num_features": settings.num_features,
        "num_vertices": settings.num_vertices,
        "num_training_frames": settings.num_training_frames,
        "training_loss": settings.training_loss,
        "max_vertex_error": settings.max_vertex_error,
    }
    return fmt.save_model(directory, settings.model_name,
                          "linear" if bridge.ACTIVE_KIND == "linear" else "neural",
                          bridge.ACTIVE_SPEC, arrays, bridge._MORPH_NAMES, stats)


def import_model(settings):
    if not settings.model_dir:
        raise ValueError("Set a Model Directory in Preview & IO first")
    data = fmt.load_model(settings.model_dir)
    spec = data["spec"]
    if data["model_kind"] == "linear":
        model = LinearRegressor()
        model.matrix = data["arrays"]["matrix"]
        kind = "linear"
        morph_deltas = None
    elif data["model_kind"] == "neural":
        morph_deltas = data["arrays"]["morph_deltas"]
        hidden = [int(h) for h in data["arrays"]["hidden_sizes"]]
        model = NeuralMorphRegressor(spec.num_features, morph_deltas, hidden,
                                     activation="relu", params=data["arrays"])
        kind = "neural"
    else:
        raise ValueError("unknown model kind %r" % data["model_kind"])
    bridge.ACTIVE_MODEL = model
    bridge.ACTIVE_SPEC = spec
    bridge.ACTIVE_INPUT = (lambda rot, trans, scales, curves:
                           spec.build_vector(rot, trans, scales, curves))
    bridge.ACTIVE_MORPH_DELTAS = morph_deltas
    bridge.ACTIVE_KIND = kind
    bridge._MORPH_NAMES = list(data["morph_names"])
    settings.model_kind = "LINEAR" if kind == "linear" else "NEURAL"
    stats = data["stats"]
    settings.is_trained = True
    settings.num_features = int(stats.get("num_features", spec.num_features))
    settings.num_vertices = int(stats.get("num_vertices", 0))
    settings.num_training_frames = int(stats.get("num_training_frames", 0))
    settings.training_loss = float(stats.get("training_loss", 0.0))
    settings.max_vertex_error = float(stats.get("max_vertex_error", 0.0))
    return data["name"]
