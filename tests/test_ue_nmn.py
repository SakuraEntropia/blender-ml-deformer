# Copyright (c) 2026 PoseDeformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Tests for the engine-format layer (nmn/ubnne) and the ONNX reader.

The hand-crafted byte layouts below are built independently of the writer
code, straight from the documented binary layout, so reader and writer are
validated against a neutral reference.
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from core.ue_nmn import (
    NmnModel, NmnError, build_nmn, parse_nmn, build_ubnne, parse_ubnne,
    run_layers, run_nmn, build_ue_input, quat_to_six_floats,
    axis_angle_to_quat, linear_sequence, extract_mlp,
    NMN_MAGIC, UBNNE_MAGIC, MODE_GLOBAL,
)


# ---------------------------------------------------------------------------
# Independent hand-rolled builders
# ---------------------------------------------------------------------------

def _align(n, a):
    return (n + a - 1) // a * a


def _u32(v):
    return struct.pack("<I", v)


def _pad(buf, a):
    while len(buf) % a:
        buf += b"\x00"
    return buf


def hand_ubnne(linear_layers, activation_id, activation_size):
    """linear_layers: [(in, out, weights(in,out), biases(out)), ...]"""
    buf = bytearray()
    buf += _u32(UBNNE_MAGIC)
    buf += _u32(1)
    buf = _pad(buf, 4)
    buf += _u32(1)  # sequence
    buf = _pad(buf, 4)
    buf += _u32(len(linear_layers) * 2)
    for in_size, out_size, weights, biases in linear_layers:
        buf = _pad(buf, 4)
        buf += _u32(4)  # linear
        buf = _pad(buf, 4)
        buf += _u32(in_size)
        buf = _pad(buf, 4)
        buf += _u32(out_size)
        buf = _pad(buf, 64)
        buf += np.asarray(biases, dtype="<f4").tobytes()
        buf = _pad(buf, 64)
        buf += np.asarray(weights, dtype="<f4").tobytes()
        buf = _pad(buf, 4)
        buf += _u32(activation_id)
        buf = _pad(buf, 4)
        buf += _u32(activation_size)
    return bytes(buf)


def hand_nmn(num_morphs, num_bones, num_curves, means, stds, main_blob):
    buf = bytearray()
    buf += _u32(NMN_MAGIC)
    buf += _u32(1)
    for v in (1, num_morphs, 0, num_bones, num_curves, 0, 0, 1):
        buf = _pad(buf, 4)
        buf += _u32(v)
    buf = _pad(buf, 64)
    buf += np.asarray(means, dtype="<f4").tobytes()
    buf = _pad(buf, 64)
    buf += np.asarray(stds, dtype="<f4").tobytes()
    runtime = b"NNERuntimeBasicCpu"
    buf = _pad(buf, 4)
    buf += _u32(len(runtime))
    buf += runtime
    buf = _pad(buf, 4)
    buf += _u32(len(main_blob))
    buf = _pad(buf, 64)
    buf += main_blob
    return bytes(buf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ubnne_reader_against_handcrafted_bytes():
    W = np.arange(12, dtype=np.float32).reshape(3, 4)  # (in, out)
    b = np.array([0.5, -0.5, 1.0, 2.0], dtype=np.float32)
    blob = hand_ubnne([(3, 4, W, b), (4, 2, np.ones((4, 2), np.float32),
                                     np.zeros(2, np.float32))], 8, 4)
    root = parse_ubnne(blob)
    assert root["type"] == "sequence"
    kids = root["children"]
    assert [k["type"] for k in kids] == ["linear", "elu", "linear", "elu"]
    assert kids[0]["in"] == 3 and kids[0]["out"] == 4
    assert np.allclose(kids[0]["weights"], W)
    assert np.allclose(kids[0]["biases"], b)
    assert kids[1]["size"] == 4
    x = np.array([[1.0, 2.0, 3.0]])
    y = run_layers(root, x)
    assert y.shape == (1, 2)


def test_ubnne_writer_reader_roundtrip():
    rng = np.random.default_rng(0)
    root = linear_sequence([8, 4], [
        rng.normal(size=(8, 5)), rng.normal(size=(4, 8)), rng.normal(size=(2, 4))],
        [np.zeros(8), np.zeros(4), np.zeros(2)])
    blob = build_ubnne(root)
    parsed = parse_ubnne(blob)
    kids = parsed["children"]
    assert kids[0]["type"] == "linear"
    assert kids[0]["weights"].shape == (5, 8)
    # alignments: blob size must be a multiple of 4 at least
    assert len(blob) % 4 == 0
    x = rng.normal(size=(3, 5))
    assert np.allclose(run_layers(root, x), run_layers(parsed, x))


def test_nmn_reader_against_handcrafted_bytes():
    num_bones, num_curves, num_morphs = 2, 1, 3
    n_inputs = num_bones * 6 + num_curves
    means = np.arange(n_inputs, dtype=np.float32) * 0.01
    stds = np.ones(n_inputs, dtype=np.float32)
    W = np.ones((n_inputs, num_morphs), dtype=np.float32)
    blob = hand_ubnne([(n_inputs, num_morphs, W, np.zeros(num_morphs, np.float32))],
                      8, num_morphs)
    data = hand_nmn(num_morphs, num_bones, num_curves, means, stds, blob)
    model = parse_nmn(data)
    assert model.num_bones == num_bones
    assert model.num_curves == num_curves
    assert model.num_morphs == num_morphs
    assert model.mode == MODE_GLOBAL
    assert model.num_main_inputs() == n_inputs
    assert np.allclose(model.input_mean, means)
    assert np.allclose(model.input_std, stds)
    assert model.runtime_name == "NNERuntimeBasicCpu"


def test_nmn_writer_reader_roundtrip():
    rng = np.random.default_rng(1)
    model = NmnModel()
    model.mode = MODE_GLOBAL
    model.num_morphs = 5
    model.num_bones = 3
    model.num_curves = 2
    model.num_floats_per_curve = 1
    n = model.num_main_inputs()
    model.input_mean = rng.normal(size=n)
    model.input_std = rng.uniform(0.5, 2.0, size=n)
    model.main_layers = linear_sequence(
        [16, 8],
        [rng.normal(size=(16, n)), rng.normal(size=(8, 16)), rng.normal(size=(5, 8))],
        [np.zeros(16), np.zeros(8), np.zeros(5)])
    data = build_nmn(model)
    parsed = parse_nmn(data)
    assert parsed.num_bones == 3 and parsed.num_curves == 2
    assert parsed.num_morphs == 5
    assert np.allclose(parsed.input_mean, model.input_mean)
    assert np.allclose(parsed.input_std, model.input_std)
    x = rng.normal(size=(4, n))
    assert np.allclose(run_layers(model.main_layers, x),
                       run_layers(parsed.main_layers, x))


def test_quat_to_six_floats_identity_and_rotation():
    assert np.allclose(quat_to_six_floats((0.0, 0.0, 0.0, 1.0)),
                       [1, 0, 0, 0, 1, 0])
    # +90 deg about Z, mirroring the engine's exact formula
    s = np.sqrt(0.5)
    out = quat_to_six_floats((0.0, 0.0, s, s))
    assert np.allclose(out, [0, -1, 0, 1, 0, 0], atol=1e-6)


def test_build_ue_input_layout():
    rotations = np.zeros((2, 3))       # two bones at rest
    curves = np.array([0.25, 0.75])
    x = build_ue_input(rotations, curves, num_curves=2, local_mode=False)
    assert x.shape == (2 * 6 + 2,)
    assert np.allclose(x[:12], [1, 0, 0, 0, 1, 0] * 2)
    assert np.allclose(x[12:], curves)


def test_run_nmn_standardizes_and_clamps():
    model = NmnModel()
    model.mode = MODE_GLOBAL
    model.num_morphs = 2
    model.num_bones = 1
    model.num_curves = 0
    n = 6
    model.input_mean = np.zeros(n)
    model.input_std = np.ones(n)
    W = np.array([[1.0, 0.0], [0.0, 1.0], [0, 0], [0, 0], [0, 0], [0, 0]])
    model.main_layers = linear_sequence([2], [W.T], [np.zeros(2)])
    x = np.array([2.0, -3.0, 0, 0, 0, 0])
    weights = run_nmn(model, x)
    # the engine layout applies ELU after the output layer too; weights are
    # then clamped to [0, 1] (2 -> 1)
    assert np.allclose(weights, [1.0, 0.0])
    weights = run_nmn(model, x, clamp=False)
    assert np.allclose(weights, [2.0, -0.9502], atol=1e-4)


def test_extract_mlp_roundtrip():
    rng = np.random.default_rng(2)
    hidden = [10, 6]
    n_in, n_out = 12, 3
    weights = [rng.normal(size=(hidden[0], n_in)),
               rng.normal(size=(hidden[1], hidden[0])),
               rng.normal(size=(n_out, hidden[1]))]
    biases = [np.zeros(hidden[0]), np.zeros(hidden[1]), np.zeros(n_out)]
    root = linear_sequence(hidden, weights, biases)
    got_hidden, got_weights, got_biases, activation = extract_mlp(root)
    assert got_hidden == hidden
    assert activation == "elu"
    for gw, w in zip(got_weights, weights):
        assert np.allclose(gw, w)
    x = rng.normal(size=(2, n_in))
    assert np.allclose(run_layers(root, x), _mlp_forward(weights, biases, x))


def _mlp_forward(weights, biases, x):
    for k, (w, b) in enumerate(zip(weights, biases)):
        x = x @ w.T + b
        x = np.where(x > 0.0, x, np.exp(x) - 1.0)  # ELU after every layer
    return x


def test_unsupported_layer_raises():
    blob = hand_ubnne([], 0, 0)
    with pytest.raises(NmnError):
        parse_ubnne(b"\x00" * 128)


def test_axis_angle_quat_conversion():
    v = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    q = axis_angle_to_quat(v)
    assert q.shape == (2, 4)
    assert np.allclose(q[0], [np.sin(0.25), 0, 0, np.cos(0.25)])
    assert np.allclose(q[1], [0, 0, 0, 1])


# ---------------------------------------------------------------------------
# ONNX: hand-rolled protobuf for a Gemm+Relu graph
# ---------------------------------------------------------------------------

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


def hand_onnx_gemm_relu():
    W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="<f4")  # (in,out)
    B = np.array([0.5, -0.5], dtype="<f4")
    node1 = (_str_field(1, "x") + _str_field(1, "W") + _str_field(1, "B")
             + _str_field(2, "y") + _str_field(4, "Gemm"))
    node2 = (_str_field(1, "y") + _str_field(2, "z") + _str_field(4, "Relu"))
    graph = (_len_field(1, node1) + _len_field(1, node2)
             + _len_field(5, _tensor("W", W)) + _len_field(5, _tensor("B", B))
             + _len_field(11, _str_field(1, "x"))
             + _len_field(12, _str_field(1, "z")))
    return _len_field(7, graph)


def test_onnx_gemm_relu():
    from core.onnx_io import parse_onnx, run_onnx
    graph = parse_onnx(hand_onnx_gemm_relu())
    assert graph["inputs"] == ["x"]
    assert graph["outputs"] == ["z"]
    x = np.array([1.0, 2.0, 3.0])
    y = x @ np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) + np.array([0.5, -0.5])
    out = run_onnx(graph, x, clamp_outputs=None)
    assert np.allclose(out, np.maximum(y, 0.0))
