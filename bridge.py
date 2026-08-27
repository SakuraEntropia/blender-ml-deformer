# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Blender-bound primitives: pose/mesh evaluation, shape keys, preview proxy,
baking, and the shared runtime state. All bpy imports live here and in the
other top-level modules; ``core`` stays bpy-free."""

from __future__ import annotations

import json
import math
import os
import time

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .core.features import BoneFeature, FeatureSpec
from .core.regressor import LinearRegressor
from .core.network import NeuralMorphRegressor

# ---------------------------------------------------------------------------
# Runtime state (cleared on load / clear)
# ---------------------------------------------------------------------------
ACTIVE_MODEL = None        # predictor: LinearRegressor | NeuralMorphRegressor | _UePredictor
ACTIVE_INPUT = None        # callable(rot, trans, scales, curves) -> feature vector
ACTIVE_SPEC = None         # FeatureSpec (for pose-format models)
ACTIVE_MORPH_DELTAS = None # (3V, M) or None
ACTIVE_KIND = None         # "pose" | "ue"
TRAINING_CACHE = None      # dict with X, D, Y, rotations, curves, base, spec, ...
_MORPH_NAMES = []

_LAST_REFRESH_TIME = 0.0
_REFRESH_INTERVAL = 0.1
_REFRESHING = False

BAKE_KEY_PREFIX = "BMD_"


def clear_runtime():
    global ACTIVE_MODEL, ACTIVE_INPUT, ACTIVE_SPEC, ACTIVE_MORPH_DELTAS
    global ACTIVE_KIND, TRAINING_CACHE, _MORPH_NAMES
    ACTIVE_MODEL = None
    ACTIVE_INPUT = None
    ACTIVE_SPEC = None
    ACTIVE_MORPH_DELTAS = None
    ACTIVE_KIND = None
    TRAINING_CACHE = None
    _MORPH_NAMES = []


def get_depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def get_settings(scene=None):
    if scene is None:
        scene = bpy.context.scene
    return scene.bmd


# ---------------------------------------------------------------------------
# Pose helpers (axis-angle local rotations)
# ---------------------------------------------------------------------------

def _matrix_basis(rot_vec, trans, scale):
    angle = math.sqrt(rot_vec[0] ** 2 + rot_vec[1] ** 2 + rot_vec[2] ** 2)
    if angle > 1e-12:
        axis = Vector(rot_vec) / angle
    else:
        axis = Vector((1.0, 0.0, 0.0))
        angle = 0.0
    half = angle * 0.5
    q = Quaternion((math.cos(half),
                    axis[0] * math.sin(half),
                    axis[1] * math.sin(half),
                    axis[2] * math.sin(half)))
    m = Matrix.Translation(trans) @ q.to_matrix().to_4x4()
    return m @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))


def read_pose(armature, spec):
    """Current pose -> (rotations (B,3) axis-angle, translations, scales)."""
    B = len(spec.bones)
    rotations = np.zeros((B, 3))
    translations = np.zeros((B, 3))
    scales = np.zeros((B, 3))
    pose_bones = armature.pose.bones
    for i, bone in enumerate(spec.bones):
        pb = pose_bones.get(bone.name)
        if pb is None:
            continue
        m = pb.matrix_basis
        axis, angle = m.to_quaternion().to_axis_angle()  # 5.x: (axis, angle)
        rotations[i] = (axis[0] * angle, axis[1] * angle, axis[2] * angle)
        t = m.to_translation()
        translations[i] = (t.x, t.y, t.z)
        s = m.to_scale()
        scales[i] = (s.x - 1.0, s.y - 1.0, s.z - 1.0)
    return rotations, translations, scales


def write_pose(armature, spec, rotations, translations, scales):
    """Apply sampled arrays (or the bind pose when rotations is None)."""
    pose_bones = armature.pose.bones
    for i, bone in enumerate(spec.bones):
        pb = pose_bones.get(bone.name)
        if pb is None:
            continue
        if rotations is None:
            pb.matrix_basis = Matrix.Identity(4).copy()
            continue
        r, t, s = rotations[i], translations[i], scales[i]
        pb.matrix_basis = _matrix_basis((r[0], r[1], r[2]),
                                        (t[0], t[1], t[2]),
                                        (1.0 + s[0], 1.0 + s[1], 1.0 + s[2]))


def snapshot_pose(armature):
    return {pb.name: pb.matrix_basis.copy() for pb in armature.pose.bones}


def restore_pose(armature, snapshot):
    for name, m in snapshot.items():
        pb = armature.pose.bones.get(name)
        if pb is not None:
            pb.matrix_basis = m


# ---------------------------------------------------------------------------
# Shape keys
# ---------------------------------------------------------------------------

def _key_blocks(mesh_obj):
    sk = mesh_obj.data.shape_keys
    return sk.key_blocks if sk is not None else None


def read_key_values(mesh_obj, names):
    kbs = _key_blocks(mesh_obj)
    if kbs is None or not names:
        return np.zeros(0)
    return np.array([kbs[n].value if n in kbs else 0.0 for n in names])


def set_key_values(mesh_obj, names, values):
    if not names:
        return
    kbs = _key_blocks(mesh_obj)
    if kbs is None:
        return
    for n, v in zip(names, values):
        kb = kbs.get(n)
        if kb is not None:
            kb.value = float(v)


def snapshot_keys(mesh_obj):
    kbs = _key_blocks(mesh_obj)
    return None if kbs is None else {kb.name: kb.value for kb in kbs}


def restore_keys(mesh_obj, snapshot):
    if snapshot is None:
        return
    kbs = _key_blocks(mesh_obj)
    if kbs is None:
        return
    for name, value in snapshot.items():
        kb = kbs.get(name)
        if kb is not None:
            kb.value = value


# ---------------------------------------------------------------------------
# Mesh evaluation
# ---------------------------------------------------------------------------

def eval_positions(mesh_obj, depsgraph):
    ob_eval = mesh_obj.evaluated_get(depsgraph)
    me = ob_eval.to_mesh(depsgraph=depsgraph)
    try:
        n = len(me.vertices)
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        return co.reshape(-1, 3)
    finally:
        ob_eval.to_mesh_clear()


def _managed_keys(settings):
    return ([e.name for e in settings.curve_inputs if e.use]
            + [e.name for e in settings.morph_targets if e.use])


def capture_base(settings, spec, depsgraph):
    """Source mesh at bind pose with managed keys at zero (restored after)."""
    mesh = settings.mesh
    armature = settings.armature
    pose_snap = snapshot_pose(armature) if armature is not None else None
    key_snap = snapshot_keys(mesh)
    managed = _managed_keys(settings)
    try:
        if armature is not None:
            write_pose(armature, spec, None, None, None)
        set_key_values(mesh, managed, [0.0] * len(managed))
        depsgraph.update()
        return eval_positions(mesh, depsgraph)
    finally:
        if pose_snap is not None:
            restore_pose(armature, pose_snap)
        restore_keys(mesh, key_snap)
        depsgraph.update()


def capture_posed_base(settings, depsgraph):
    """Source mesh at the current pose with managed keys at zero."""
    mesh = settings.mesh
    key_snap = snapshot_keys(mesh)
    managed = _managed_keys(settings)
    try:
        set_key_values(mesh, managed, [0.0] * len(managed))
        depsgraph.update()
        return eval_positions(mesh, depsgraph)
    finally:
        restore_keys(mesh, key_snap)
        depsgraph.update()


def compute_morph_deltas(mesh_obj, morph_names):
    """(3V, M) per-vertex offsets of the morph shape keys."""
    sk = mesh_obj.data.shape_keys
    if sk is None:
        return np.zeros((0, len(morph_names)))
    kbs = sk.key_blocks
    basis = kbs[0]
    V = len(mesh_obj.data.vertices)
    basis_co = np.empty(V * 3)
    basis.data.foreach_get("co", basis_co)
    out = np.zeros((V * 3, len(morph_names)))
    rel_co = np.empty(V * 3)
    for j, name in enumerate(morph_names):
        kb = kbs.get(name)
        if kb is None:
            continue
        co = np.empty(V * 3)
        kb.data.foreach_get("co", co)
        if kb.relative_key == kb:
            out[:, j] = co - basis_co
        else:
            kb.relative_key.data.foreach_get("co", rel_co)
            out[:, j] = co - rel_co
    return out


# ---------------------------------------------------------------------------
# Spec / list building
# ---------------------------------------------------------------------------

def build_spec(settings):
    spec = FeatureSpec()
    for entry in settings.bones:
        if not (entry.use_rotation or entry.use_translation or entry.use_scale):
            continue
        spec.bones.append(BoneFeature(
            entry.name,
            use_rotation=entry.use_rotation,
            use_translation=entry.use_translation,
            use_scale=entry.use_scale,
            rotation_axes=(entry.use_rotation_x, entry.use_rotation_y,
                           entry.use_rotation_z)))
    spec.curve_names = [e.name for e in settings.curve_inputs if e.use]
    spec.rotation_weight = settings.input_rotation_weight
    spec.translation_weight = settings.input_translation_weight
    spec.scale_weight = settings.input_scale_weight
    spec.curve_weight = settings.input_curve_weight
    return spec


def sync_bones(settings):
    armature = settings.armature
    if armature is None:
        return 0
    preserved = {e.name: e for e in settings.bones}
    settings.bones.clear()
    for bone in armature.data.bones:
        e = settings.bones.add()
        e.name = bone.name
        if bone.name in preserved:
            src = preserved[bone.name]
            for attr in ("use_rotation", "use_rotation_x", "use_rotation_y",
                         "use_rotation_z", "rotation_range_x", "rotation_range_y",
                         "rotation_range_z", "use_translation", "use_scale"):
                setattr(e, attr, getattr(src, attr))
    return len(settings.bones)


def _sync_keys(settings, target):
    mesh = settings.mesh
    if mesh is None or mesh.data.shape_keys is None:
        target.clear()
        return 0
    preserved = {e.name: e.use for e in target}
    target.clear()
    for kb in mesh.data.shape_keys.key_blocks[1:]:
        e = target.add()
        e.name = kb.name
        e.use = preserved.get(kb.name, False)
    return len(target)


def sync_curve_inputs(settings):
    return _sync_keys(settings, settings.curve_inputs)


def sync_morph_targets(settings):
    return _sync_keys(settings, settings.morph_targets)


def rotation_ranges(settings):
    return {e.name: (e.rotation_range_x, e.rotation_range_y, e.rotation_range_z)
            for e in settings.bones if e.use_rotation}


def parse_hidden_layers(text):
    sizes = []
    for part in str(text).split(","):
        part = part.strip()
        if part.isdigit() and int(part) > 0:
            sizes.append(int(part))
    return sizes if sizes else [32]


# ---------------------------------------------------------------------------
# Predictor plumbing
# ---------------------------------------------------------------------------

def pose_feature_vector(settings, spec, depsgraph):
    rotations, translations, scales = read_pose(settings.armature, spec)
    curves = read_key_values(settings.mesh, spec.curve_names)
    return spec.build_vector(rotations, translations, scales, curves)


def current_feature_vector(settings, depsgraph):
    """Build the input for the ACTIVE predictor from the current scene state."""
    if ACTIVE_INPUT is None:
        return None
    rotations, translations, scales = read_pose(settings.armature, ACTIVE_SPEC)
    curves = read_key_values(settings.mesh, ACTIVE_SPEC.curve_names)
    return ACTIVE_INPUT(rotations, translations, scales, curves)


def model_predict(x):
    """Deltas (3V, N) from the active predictor for a feature matrix (F, N)."""
    if ACTIVE_MODEL is None:
        raise RuntimeError("no model loaded")
    return ACTIVE_MODEL.predict(x)


def uses_posed_base():
    """Neural-morph style models only add corrective deltas on top of the
    posed (skinned) mesh; linear pose models contain the full deformation."""
    return ACTIVE_KIND == "neural" or ACTIVE_KIND == "ue"


# ---------------------------------------------------------------------------
# Preview proxy
# ---------------------------------------------------------------------------

def create_preview_proxy(context):
    settings = context.scene.bmd
    src = settings.mesh
    if src is None:
        raise ValueError("Pick a Mesh in the Setup section first")
    depsgraph = get_depsgraph()
    spec = build_spec(settings)
    base = (capture_base(settings, spec, depsgraph)
            if (ACTIVE_MODEL is not None and ACTIVE_INPUT is not None)
            else eval_positions(src, depsgraph))
    src_eval = src.evaluated_get(depsgraph)
    me = src_eval.to_mesh(depsgraph=depsgraph)
    try:
        new_me = bpy.data.meshes.new_from_object(src_eval)
    finally:
        src_eval.to_mesh_clear()
    if new_me is None or len(new_me.vertices) != len(base):
        new_me = bpy.data.meshes.new(src.name + "_BMDPreview")
        new_me.vertices.add(len(base))
    if new_me.shape_keys is not None:
        bpy.data.shape_keys.remove(new_me.shape_keys, do_unlink=True)
    new_me.name = src.name + "_BMDPreview"
    new_me.vertices.foreach_set("co", base.ravel())
    new_me.update()
    obj = bpy.data.objects.new(src.name + "_BMDPreview", new_me)
    context.scene.collection.objects.link(obj)
    settings.preview_object = obj
    refresh_preview(context.scene, force=True)
    return obj


def refresh_preview(scene, force=False):
    global _LAST_REFRESH_TIME, _REFRESHING
    settings = scene.bmd
    if ACTIVE_MODEL is None or ACTIVE_INPUT is None:
        return
    if _REFRESHING:
        return
    if not force and time.monotonic() - _LAST_REFRESH_TIME < _REFRESH_INTERVAL:
        return
    proxy = settings.preview_object
    if proxy is None or settings.armature is None or settings.mesh is None:
        return
    _REFRESHING = True
    try:
        depsgraph = get_depsgraph()
        x = current_feature_vector(settings, depsgraph)
        if x is None:
            return
        delta = model_predict(x[:, None])[:, 0]
        if uses_posed_base():
            base = capture_posed_base(settings, depsgraph)
        else:
            base = capture_base(settings, ACTIVE_SPEC, depsgraph)
        me = proxy.data
        if len(me.vertices) != base.shape[0]:
            return
        me.vertices.foreach_set("co", (base + delta.reshape(-1, 3)).ravel())
        me.update()
        _LAST_REFRESH_TIME = time.monotonic()
    finally:
        _REFRESHING = False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _try_refresh(scene, force=False):
    try:
        refresh_preview(scene, force=force)
    except Exception as exc:
        print("[Blender ML Deformer] preview refresh failed:", exc)


@bpy.app.handlers.persistent
def _depsgraph_handler(scene, depsgraph):
    if scene.bmd.auto_refresh:
        _try_refresh(scene)


@bpy.app.handlers.persistent
def _frame_change_handler(scene):
    _try_refresh(scene, force=True)


@bpy.app.handlers.persistent
def _load_handler(_unused):
    clear_runtime()
    try:
        scene = bpy.context.scene
        if scene is None:
            return
        settings = scene.bmd
        if settings.is_trained and settings.model_dir:
            from .train import import_model
            if os.path.isfile(os.path.join(settings.model_dir, "pose_model.json")):
                import_model(settings)
                print("[Blender ML Deformer] re-imported model from %s" % settings.model_dir)
    except Exception as exc:
        print("[Blender ML Deformer] auto-import failed:", exc)


def register_handlers():
    bpy.app.handlers.depsgraph_update_post.append(_depsgraph_handler)
    bpy.app.handlers.frame_change_post.append(_frame_change_handler)
    bpy.app.handlers.load_post.append(_load_handler)


def unregister_handlers():
    for handler, key in ((_depsgraph_handler, bpy.app.handlers.depsgraph_update_post),
                         (_frame_change_handler, bpy.app.handlers.frame_change_post),
                         (_load_handler, bpy.app.handlers.load_post)):
        if handler in key:
            key.remove(handler)
