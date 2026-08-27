# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Reader / writer / executor for the engine-side neural morph network
exchange format (the ``*.nmn`` file and its embedded "ubnne" bytecode model).

The binary layout implemented here is derived from the publicly documented
behaviour of the engine runtime:

* outer file: uint32 magic 0x234A1304, uint32 version 1, then nine uint32
  header fields (mode, morph counts, bone/curve/group counts), 64-byte
  aligned float arrays of input means / stds, a length-prefixed runtime name
  string, and 64-byte aligned embedded model blobs;
* embedded model: uint32 magic 0x0BA51C01, uint32 version 1, then a layer
  tree (Sequence=1, Linear=4, ReLU=7, ELU=8, TanH=9, GELU=20, ...) where
  uint32 scalars are 4-byte aligned, float arrays are 64-byte aligned, and
  linear weights are stored row-major as [input][output];
* bone inputs are 6 floats per bone: columns 0 and 1 of the 3x3 matrix of
  the bone's local rotation (relative to its parent); curve inputs are raw
  values; the vector is standardized with the stored means/stds before the
  model runs.

Only the layer kinds an MLP can contain are executed; anything else raises
a clear error.
"""

from __future__ import annotations

import struct

import numpy as np

NMN_MAGIC = 0x234A1304
NMN_VERSION = 1
UBNNE_MAGIC = 0x0BA51C01
UBNNE_VERSION = 1

MODE_LOCAL = 0
MODE_GLOBAL = 1

LAYER_SEQUENCE = 1
LAYER_NORMALIZE = 2
LAYER_DENORMALIZE = 3
LAYER_LINEAR = 4
LAYER_COMPRESSED_LINEAR = 5
LAYER_MULTILINEAR = 6
LAYER_RELU = 7
LAYER_ELU = 8
LAYER_TANH = 9
LAYER_PRELU = 10
LAYER_MEMORY_CELL = 11
LAYER_COPY = 12
LAYER_CONCAT = 13
LAYER_ARRAY = 14
LAYER_AGGREGATE_SET = 15
LAYER_AGGREGATE_OR_EXCLUSIVE = 16
LAYER_AGGREGATE_OR_INCLUSIVE = 17
LAYER_CLAMP = 18
LAYER_SPARSE_MIXTURE_OF_EXPERTS = 19
LAYER_GELU = 20

_LAYER_NAMES = {
    LAYER_SEQUENCE: "sequence", LAYER_LINEAR: "linear", LAYER_RELU: "relu",
    LAYER_ELU: "elu", LAYER_TANH: "tanh", LAYER_GELU: "gelu",
    LAYER_NORMALIZE: "normalize", LAYER_DENORMALIZE: "denormalize",
    LAYER_COMPRESSED_LINEAR: "compressed_linear", LAYER_MULTILINEAR: "multilinear",
    LAYER_PRELU: "prelu", LAYER_MEMORY_CELL: "memory_cell", LAYER_COPY: "copy",
    LAYER_CONCAT: "concat", LAYER_ARRAY: "array", LAYER_AGGREGATE_SET: "aggregate_set",
    LAYER_AGGREGATE_OR_EXCLUSIVE: "aggregate_or_exclusive",
    LAYER_AGGREGATE_OR_INCLUSIVE: "aggregate_or_inclusive", LAYER_CLAMP: "clamp",
    LAYER_SPARSE_MIXTURE_OF_EXPERTS: "sparse_mixture_of_experts",
}

DEFAULT_RUNTIME_NAME = "NNERuntimeBasicCpu"


class NmnError(Exception):
    pass


class NmnModel:
    """Parsed neural morph network (plus embedded model layers)."""

    def __init__(self):
        self.mode = MODE_GLOBAL
        self.num_morphs = 0
        self.num_morphs_per_bone = 0
        self.num_bones = 0
        self.num_curves = 0
        self.num_groups = 0
        self.num_items_per_group = 0
        self.num_floats_per_curve = 1
        self.input_mean = np.zeros(0)
        self.input_std = np.ones(0)
        self.runtime_name = DEFAULT_RUNTIME_NAME
        self.main_layers = {"type": "sequence", "children": []}
        self.group_layers = None

    @property
    def is_local_mode(self):
        return self.mode == MODE_LOCAL

    def num_main_inputs(self):
        per_curve = 6 if self.is_local_mode else 1
        return self.num_bones * 6 + self.num_curves * per_curve


# ---------------------------------------------------------------------------
# Byte-level helpers (alignment rules of the engine serialization)
# ---------------------------------------------------------------------------

def _align(offset, alignment):
    return (offset + alignment - 1) // alignment * alignment


class _Reader:
    def __init__(self, data):
        self.data = data
        self.off = 0

    def u32(self):
        self.off = _align(self.off, 4)
        if self.off + 4 > len(self.data):
            raise NmnError("unexpected end of file at offset %d" % self.off)
        (v,) = struct.unpack_from("<I", self.data, self.off)
        self.off += 4
        return v

    def f32s(self, count):
        self.off = _align(self.off, 64)
        nbytes = count * 4
        if self.off + nbytes > len(self.data):
            raise NmnError("unexpected end of file at offset %d" % self.off)
        v = np.frombuffer(self.data, dtype="<f4", count=count, offset=self.off)
        self.off += nbytes
        return v.astype(np.float64).copy()

    def raw(self, count):
        if self.off + count > len(self.data):
            raise NmnError("unexpected end of file at offset %d" % self.off)
        v = self.data[self.off:self.off + count]
        self.off += count
        return v

    def string(self):
        n = self.u32()
        return self.raw(n).decode("utf-8")


class _Writer:
    def __init__(self):
        self.buf = bytearray()

    def pad_to(self, alignment):
        while len(self.buf) % alignment:
            self.buf.append(0)

    def u32(self, value):
        self.pad_to(4)
        self.buf += struct.pack("<I", int(value))

    def f32s(self, values):
        self.pad_to(64)
        self.buf += np.asarray(values, dtype="<f4").ravel().tobytes()

    def raw(self, blob):
        self.buf += blob

    def string(self, text):
        data = text.encode("utf-8")
        self.u32(len(data))
        self.buf += data

    def bytes(self):
        return bytes(self.buf)


# ---------------------------------------------------------------------------
# Embedded model (ubnne) parsing / building / execution
# ---------------------------------------------------------------------------

def _parse_ubnne_element(r):
    type_id = r.u32()
    name = _LAYER_NAMES.get(type_id)
    if name is None:
        raise NmnError("unsupported layer type id %d" % type_id)
    if name == "sequence":
        count = r.u32()
        children = [_parse_ubnne_element(r) for _ in range(count)]
        return {"type": "sequence", "children": children}
    if name == "linear":
        in_size = r.u32()
        out_size = r.u32()
        biases = r.f32s(out_size)
        weights = r.f32s(in_size * out_size).reshape(in_size, out_size)
        return {"type": "linear", "in": in_size, "out": out_size,
                "weights": weights, "biases": biases}
    if name in ("relu", "elu", "tanh", "gelu"):
        size = r.u32()
        return {"type": name, "size": size}
    raise NmnError("layer type %r is not supported by this reader" % name)


def parse_ubnne(data):
    r = _Reader(data)
    if r.u32() != UBNNE_MAGIC:
        raise NmnError("bad embedded model magic")
    if r.u32() != UBNNE_VERSION:
        raise NmnError("unsupported embedded model version")
    root = _parse_ubnne_element(r)
    if root["type"] != "sequence":
        root = {"type": "sequence", "children": [root]}
    return root


def _write_ubnne_element(w, layer):
    t = layer["type"]
    if t == "sequence":
        w.u32(LAYER_SEQUENCE)
        w.u32(len(layer["children"]))
        for child in layer["children"]:
            _write_ubnne_element(w, child)
    elif t == "linear":
        w.u32(LAYER_LINEAR)
        w.u32(layer["in"])
        w.u32(layer["out"])
        w.f32s(layer["biases"])
        w.f32s(layer["weights"])
    elif t in ("relu", "elu", "tanh", "gelu"):
        w.u32({"relu": LAYER_RELU, "elu": LAYER_ELU, "tanh": LAYER_TANH,
               "gelu": LAYER_GELU}[t])
        w.u32(layer["size"])
    else:
        raise NmnError("cannot write layer type %r" % t)


def build_ubnne(root):
    w = _Writer()
    w.u32(UBNNE_MAGIC)
    w.u32(UBNNE_VERSION)
    _write_ubnne_element(w, root)
    return w.bytes()


def _apply_activation(name, x):
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "elu":
        return np.where(x > 0.0, x, np.exp(x) - 1.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    raise NmnError("cannot execute activation %r" % name)


def run_layers(layers, x):
    """x: (N, in) -> (N, out)."""
    t = layers["type"]
    if t == "sequence":
        for child in layers["children"]:
            x = run_layers(child, x)
        return x
    if t == "linear":
        if x.shape[1] != layers["in"]:
            raise NmnError("input size %d does not match layer input %d"
                           % (x.shape[1], layers["in"]))
        return x @ layers["weights"] + layers["biases"]
    if t in ("relu", "elu", "tanh", "gelu"):
        return _apply_activation(t, x)
    raise NmnError("cannot execute layer type %r" % t)


# ---------------------------------------------------------------------------
# Outer .nmn file
# ---------------------------------------------------------------------------

def parse_nmn(data):
    r = _Reader(data)
    if r.u32() != NMN_MAGIC:
        raise NmnError("not a neural morph network file (bad magic)")
    if r.u32() != NMN_VERSION:
        raise NmnError("unsupported neural morph network version")
    model = NmnModel()
    model.mode = MODE_LOCAL if r.u32() == 0 else MODE_GLOBAL
    model.num_morphs = r.u32()
    model.num_morphs_per_bone = r.u32()
    model.num_bones = r.u32()
    model.num_curves = r.u32()
    model.num_groups = r.u32()
    model.num_items_per_group = r.u32()
    model.num_floats_per_curve = r.u32()
    n_inputs = model.num_main_inputs()
    model.input_mean = r.f32s(n_inputs)
    model.input_std = r.f32s(n_inputs)
    model.runtime_name = r.string()
    main_size = r.u32()
    r.off = _align(r.off, 64)  # the embedded blob starts 64-byte aligned
    model.main_layers = parse_ubnne(r.raw(main_size))
    if model.num_groups > 0 and model.is_local_mode:
        group_size = r.u32()
        r.off = _align(r.off, 64)
        model.group_layers = parse_ubnne(r.raw(group_size))
    return model


def build_nmn(model):
    w = _Writer()
    w.u32(NMN_MAGIC)
    w.u32(NMN_VERSION)
    w.u32(0 if model.is_local_mode else 1)
    w.u32(model.num_morphs)
    w.u32(model.num_morphs_per_bone)
    w.u32(model.num_bones)
    w.u32(model.num_curves)
    w.u32(model.num_groups)
    w.u32(model.num_items_per_group)
    w.u32(model.num_floats_per_curve)
    n_inputs = model.num_main_inputs()
    w.f32s(model.input_mean[:n_inputs])
    w.f32s(model.input_std[:n_inputs])
    w.string(model.runtime_name)
    main_blob = build_ubnne(model.main_layers)
    w.u32(len(main_blob))
    w.pad_to(64)
    w.raw(main_blob)
    if model.group_layers is not None:
        group_blob = build_ubnne(model.group_layers)
        w.u32(len(group_blob))
        w.pad_to(64)
        w.raw(group_blob)
    return w.bytes()


# ---------------------------------------------------------------------------
# Input construction (bone rotations -> 6 floats) and inference
# ---------------------------------------------------------------------------

def quat_to_six_floats(q):
    """Rotation quaternion (x, y, z, w) -> columns 0 and 1 of its 3x3 matrix."""
    x, y, z, w = q
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return np.array([1.0 - (yy + zz), xy - wz, xz + wy,   # X column
                     xy + wz, 1.0 - (xx + zz), yz - wx])  # Y column


def axis_angle_to_quat(rot_vectors):
    """(B, 3) axis-angle vectors (radians) -> (B, 4) quaternions xyzw."""
    rot_vectors = np.asarray(rot_vectors, dtype=np.float64)
    angles = np.linalg.norm(rot_vectors, axis=1)
    quats = np.zeros((len(rot_vectors), 4))
    quats[:, 3] = 1.0
    nz = angles > 1e-12
    if nz.any():
        axes = rot_vectors[nz] / angles[nz, None]
        half = 0.5 * angles[nz]
        quats[nz, :3] = axes * np.sin(half)[:, None]
        quats[nz, 3] = np.cos(half)
    return quats


def build_ue_input(rot_vectors, curve_values, num_curves, local_mode=False):
    """Engine-format raw input vector from axis-angle rotations + curve values.

    rot_vectors: (B, 3) radians. curve_values: (C,) raw values.
    Returns the unstandardized vector (means/stds are applied by run_nmn).
    """
    rot_vectors = np.asarray(rot_vectors, dtype=np.float64)
    curve_values = np.asarray(curve_values, dtype=np.float64).ravel()
    if len(curve_values) < num_curves:
        curve_values = np.pad(curve_values, (0, num_curves - len(curve_values)))
    quats = axis_angle_to_quat(rot_vectors)
    six = np.stack([quat_to_six_floats(q) for q in quats]).ravel()
    if local_mode:
        # local mode stores 6 floats per curve as well
        curve_parts = [np.full(6, v) for v in curve_values[:num_curves]]
        curves = np.concatenate(curve_parts) if curve_parts else np.empty(0)
    else:
        curves = curve_values[:num_curves]
    return np.concatenate([six, curves])


def run_nmn(model, x, clamp=True):
    """Standardize, run the main network, return morph weights.

    x may be a single vector (F,) or a batch (F, N); the result is (M,) or
    (M, N) respectively."""
    if model.is_local_mode and model.num_groups > 0:
        raise NmnError("local mode with group networks is not supported by "
                       "this runtime; only global mode networks can run")
    x = np.asarray(x, dtype=np.float64)
    n = model.num_main_inputs()
    single = x.ndim == 1
    if single:
        if x.shape[0] != n:
            raise NmnError("input has %d values, network expects %d"
                           % (x.shape[0], n))
        x = x[:, None]
    if x.shape[0] != n:
        raise NmnError("input has %d features, network expects %d"
                       % (x.shape[0], n))
    x = (x - model.input_mean[:, None]) / model.input_std[:, None]
    weights = run_layers(model.main_layers, x.T).T  # (M, N)
    if clamp:
        weights = np.clip(weights, 0.0, 1.0)
    return weights[:, 0] if single else weights


# ---------------------------------------------------------------------------
# Conversion to/from Blender ML Deformer's own regressor
# ---------------------------------------------------------------------------

def linear_sequence(hidden_sizes, weights, biases, activation="elu"):
    """Build a Sequence[Linear, ELU, ..., Linear, ELU] layer tree (the layout
    the engine runtime itself produces for an MLP) from Blender ML Deformer-style
    parameters.  weights/biases: lists of (out, in) / (out,) arrays."""
    children = []
    for k, (w, b) in enumerate(zip(weights, biases)):
        w = np.asarray(w, dtype=np.float64)  # (out, in)
        b = np.asarray(b, dtype=np.float64)
        children.append({"type": "linear", "in": w.shape[1], "out": w.shape[0],
                         "weights": w.T.copy(), "biases": b.copy()})
        children.append({"type": activation, "size": w.shape[0]})
    return {"type": "sequence", "children": children}


def extract_mlp(layers):
    """Walk a layer tree and return (hidden_sizes, weights, biases, activation)
    for a plain Linear/activation MLP.  Raises if the tree does not match."""
    seq = layers
    if seq["type"] != "sequence":
        raise NmnError("top level of the model must be a sequence")
    weights, biases, activations = [], [], []
    pending_linear = None
    for child in seq["children"]:
        t = child["type"]
        if t == "linear":
            if pending_linear is not None:
                raise NmnError("two linear layers in a row are not an MLP")
            pending_linear = child
        elif t in ("relu", "elu", "tanh", "gelu"):
            if pending_linear is None:
                raise NmnError("activation without a preceding linear layer")
            weights.append(pending_linear["weights"].T.copy())  # (out, in)
            biases.append(pending_linear["biases"].copy())
            activations.append(t)
            pending_linear = None
        else:
            raise NmnError("layer type %r is not an MLP layer" % t)
    if pending_linear is not None:
        weights.append(pending_linear["weights"].T.copy())
        biases.append(pending_linear["biases"].copy())
        activations.append(activations[-1] if activations else "elu")
    if not weights:
        raise NmnError("model contains no linear layers")
    hidden_sizes = [w.shape[0] for w in weights[:-1]]
    return hidden_sizes, weights, biases, activations[-1]
