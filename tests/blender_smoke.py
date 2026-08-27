# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""End-to-end smoke test that runs inside Blender (no window needed).

Exercises the whole pipeline: sync, training data generation, both model
kinds, preview proxy, own-format export/import, shape key baking, and the
engine bridge (export .nmn -> re-import -> identical inference, plus ONNX
import of a hand-built graph).

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --factory-startup --python tests/blender_smoke.py
"""

import os
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
ADDON_DIR = str(Path(__file__).resolve().parents[1])
for p in (REPO_ROOT, ADDON_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import bpy
import numpy as np

import blender_ml_deformer as bmd
from blender_ml_deformer import bridge
from blender_ml_deformer import train
from blender_ml_deformer import ue as ue_ops
from blender_ml_deformer.core.ue_nmn import parse_nmn, run_nmn, build_ue_input
from blender_ml_deformer.core.onnx_io import parse_onnx, run_onnx

FAILURES = []


def check(cond, msg):
    if cond:
        print("  [ok] " + msg)
    else:
        FAILURES.append(msg)
        print("  [FAIL] " + msg)


def consume(gen):
    for _ in gen:
        pass


# --- tiny protobuf helpers for a hand-built ONNX file -----------------------

def _varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _len_field(num, payload):
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _str_field(num, s):
    return _len_field(num, s.encode("utf-8"))


def _i_field(num, v):
    return _varint((num << 3) | 0) + _varint(v)


def _tensor(name, arr):
    arr = np.asarray(arr, dtype="<f4")
    payload = _i_field(2, 1) + _str_field(8, name)
    payload += _len_field(9, arr.tobytes())
    for d in arr.shape:
        payload += _i_field(1, d)
    return payload


def make_onnx_file(path, num_inputs, num_morphs):
    W = np.ones((num_inputs, num_morphs), dtype="<f4")
    B = np.zeros(num_morphs, dtype="<f4")
    node = (_str_field(1, "x") + _str_field(1, "W") + _str_field(1, "B")
            + _str_field(2, "y") + _str_field(4, "Gemm"))
    graph = (_len_field(1, node)
             + _len_field(5, _tensor("W", W)) + _len_field(5, _tensor("B", B))
             + _len_field(11, _str_field(1, "x"))
             + _len_field(12, _str_field(1, "y")))
    with open(path, "wb") as f:
        f.write(_len_field(7, graph))
    return W, B


# ---------------------------------------------------------------------------

try:
    bmd.register()
    scene = bpy.context.scene

    for ob in list(scene.objects):
        bpy.data.objects.remove(ob, do_unlink=True)

    # armature: root -> child
    arm_data = bpy.data.armatures.new("BMDArm")
    arm = bpy.data.objects.new("BMDArm", arm_data)
    scene.collection.objects.link(arm)
    scene.view_layers[0].objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    root = arm_data.edit_bones.new("root")
    root.tail = (0.0, 1.0, 0.0)
    child = arm_data.edit_bones.new("child")
    child.head = (0.0, 1.0, 0.0)
    child.tail = (0.0, 2.0, 0.0)
    child.parent = root
    bpy.ops.object.mode_set(mode="OBJECT")

    # grid skinned to the child bone + one morph shape key
    verts = [(x, y, 0.0) for y in range(4) for x in range(4)]
    faces = [(iy * 4 + ix, iy * 4 + ix + 1, (iy + 1) * 4 + ix + 1,
              (iy + 1) * 4 + ix) for iy in range(3) for ix in range(3)]
    me = bpy.data.meshes.new("BMDMesh")
    me.from_pydata(verts, [], faces)
    me.update()
    obj = bpy.data.objects.new("BMDMesh", me)
    scene.collection.objects.link(obj)
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    vg = obj.vertex_groups.new(name="child")
    vg.add(range(len(verts)), 1.0, "REPLACE")
    mod.vertex_group = "child"
    obj.shape_key_add(name="Basis")
    mk = obj.shape_key_add(name="morph1", from_mix=False)
    mk.data[0].co.z += 0.5
    mk.data[5].co.z += 0.5
    mk.data.update()
    me.update()

    s = scene.bmd
    s.armature = arm
    s.mesh = obj
    s.num_random_poses = 16
    s.iterations = 400
    s.random_seed = 1234

    bpy.ops.bmd.sync_bones()
    check(len(s.bones) == 2, "sync_bones found 2 bones")

    # ---- linear model: generate -> train -> preview ----
    s.model_kind = "LINEAR"
    consume(train.generate_training_data_iter(s))
    check(s.num_training_frames == 17, "generated 16 random + 1 reference frame")
    consume(train.train_model_iter(s))
    check(s.is_trained, "linear model trained")
    check(np.isfinite(s.training_loss), "loss is finite")
    print("    linear training loss: %.6f" % s.training_loss)
    check(s.training_loss < 0.01, "linear reproduces training data (mse < 0.01)")

    bpy.ops.bmd.create_preview_proxy()
    check(s.preview_object is not None, "preview proxy created")
    base_proxy = np.array([v.co for v in s.preview_object.data.vertices])

    arm.pose.bones["child"].rotation_quaternion = (0.99875, 0.05, 0.0, 0.0)
    bpy.context.evaluated_depsgraph_get().update()
    bpy.ops.bmd.refresh_preview()
    new_proxy = np.array([v.co for v in s.preview_object.data.vertices])
    check(not np.allclose(base_proxy, new_proxy), "proxy follows the pose")
    move = float(np.linalg.norm(new_proxy - base_proxy))
    err_proxy = float(np.linalg.norm(
        new_proxy - bridge.eval_positions(obj, bridge.get_depsgraph())))
    print("    proxy move %.4f, deviation %.4f" % (move, err_proxy))
    check(err_proxy < max(0.5, 0.3 * move), "proxy captures most of the deformation")

    # ---- own-format export / import ----
    tmp = tempfile.mkdtemp(prefix="bmd_smoke_")
    s.model_dir = tmp
    s.model_name = "smoke"
    bpy.ops.bmd.export_model()
    check(os.path.isfile(os.path.join(tmp, "pose_model.json")),
          "pose_model.json written")
    check(os.path.isfile(os.path.join(tmp, "pose_model.npz")),
          "pose_model.npz written")
    bridge.clear_runtime()
    s.is_trained = False
    bpy.ops.bmd.import_model()
    check(s.is_trained and bridge.ACTIVE_MODEL is not None, "own format re-imported")

    # ---- neural model: generate -> train -> export .nmn -> re-import ----
    s.model_kind = "NEURAL"
    bpy.ops.bmd.sync_morph_targets()
    check(len(s.morph_targets) == 1, "one morph target listed")
    s.morph_targets[0].use = True
    consume(train.generate_training_data_iter(s))
    consume(train.train_model_iter(s))
    check(s.is_trained, "neural model trained")
    check(np.isfinite(s.training_loss), "neural loss is finite")
    print("    neural training loss: %.6f" % s.training_loss)

    nmn_path = os.path.join(tmp, "smoke.nmn")
    consume(ue_ops.export_nmn_iter(s, nmn_path))
    check(os.path.isfile(nmn_path), "engine network written (.nmn)")
    check(os.path.isfile(os.path.join(tmp, "smoke.bmd_ue.json")),
          "engine sidecar json written")

    # re-import the exported network through the engine path
    bridge.clear_runtime()
    s.is_trained = False
    ue_ops.import_nmn(s, nmn_path)
    check(s.is_trained and bridge.ACTIVE_MODEL is not None,
          "engine network re-imported")

    # inference consistency: bridge predictor vs direct core execution
    x = bridge.current_feature_vector(s, bridge.get_depsgraph())
    pred_bridge = bridge.model_predict(x[:, None])[:, 0]
    model2 = parse_nmn(open(nmn_path, "rb").read())
    rotations, _, _ = bridge.read_pose(arm, bridge.ACTIVE_SPEC)
    x_ue = build_ue_input(rotations, np.zeros(0), num_curves=0)
    w = run_nmn(model2, x_ue)
    morph_deltas = bridge.compute_morph_deltas(obj, ["morph1"])
    check(np.allclose(pred_bridge, morph_deltas @ w, atol=1e-5),
          "imported engine network matches direct core inference")

    # ---- bake with the imported engine network ----
    s.num_bake_poses = 4
    consume(train.bake_shape_keys_iter(s))
    baked = bpy.data.objects.get("BMDMesh_BMDBaked")
    check(baked is not None, "baked object created")
    if baked is not None:
        check(len(baked.data.shape_keys.key_blocks) == 5,
              "baked basis + 4 pose shape keys")

    # ---- ONNX import ----
    onnx_path = os.path.join(tmp, "model.onnx")
    num_inputs = 2 * 6 + 0  # 2 bones, no curves
    make_onnx_file(onnx_path, num_inputs, 1)
    bridge.clear_runtime()
    s.is_trained = False
    ue_ops.import_onnx(s, onnx_path)
    check(s.is_trained, "onnx network imported")
    graph = parse_onnx(open(onnx_path, "rb").read())
    x = bridge.current_feature_vector(s, bridge.get_depsgraph())
    pred_bridge = bridge.model_predict(x[:, None])[:, 0]
    w = run_onnx(graph, x, clamp_outputs=None)
    check(np.allclose(pred_bridge, morph_deltas @ np.clip(w, 0, 1), atol=1e-5),
          "onnx inference matches direct core execution")

    # ---- clear ----
    bpy.ops.bmd.clear()
    check(s.preview_object is None, "clear removed preview pointer")
    check(bridge.ACTIVE_MODEL is None, "clear forgot the model")

except Exception as exc:
    import traceback
    traceback.print_exc()
    FAILURES.append("exception: %r" % exc)

if FAILURES:
    print("\nSMOKE FAILED (%d failures)" % len(FAILURES))
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("\nSMOKE OK")
