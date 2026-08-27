# Copyright (c) 2026 PoseDeformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Minimal ONNX reader + pure-numpy executor.

Covers the small operator set used by plain MLP exports (Gemm/MatMul/Add/
Mul/Sub/Div, Relu/Elu/Sigmoid/Tanh/LeakyRelu/Clip, Constant/Identity/
Reshape/Flatten/Transpose/Concat/Unsqueeze/Squeeze).  This exists so pose
networks exported from the engine's python training pipeline (the UE 5.4/5.5
flow, which produced an ``*.onnx`` per model) can be loaded without the
``onnx`` / ``onnxruntime`` packages, which are not bundled with Blender.

The parser is a plain protobuf wire-format reader: no external dependencies.
"""

from __future__ import annotations

import struct

import numpy as np

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LEN = 2
WIRE_FIXED32 = 5

_TENSOR_DTYPES = {1: "<f4", 11: "<f8", 6: "<i4", 7: "<i8", 2: "u1", 3: "i1"}


class OnnxError(Exception):
    pass


def _read_varint(buf, pos, end):
    value = 0
    shift = 0
    while pos < end:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise OnnxError("truncated varint")


def _parse_fields(buf, start, end):
    """Return {field_number: [(wire_type, value), ...]}."""
    fields = {}
    pos = start
    while pos < end:
        key, pos = _read_varint(buf, pos, end)
        field, wire = key >> 3, key & 7
        if wire == WIRE_VARINT:
            value, pos = _read_varint(buf, pos, end)
        elif wire == WIRE_LEN:
            length, pos = _read_varint(buf, pos, end)
            value = buf[pos:pos + length]
            pos += length
        elif wire == WIRE_FIXED32:
            value = buf[pos:pos + 4]
            pos += 4
        elif wire == WIRE_FIXED64:
            value = buf[pos:pos + 8]
            pos += 8
        else:
            raise OnnxError("unsupported wire type %d" % wire)
        fields.setdefault(field, []).append((wire, value))
    return fields


def _msg(buf, start, end):
    if start >= end:
        return None
    return _parse_fields(buf, start, end)


def _strings(fields, number):
    return [v.decode("utf-8") for w, v in fields.get(number, []) if w == WIRE_LEN]


def _string(fields, number, default=None):
    out = _strings(fields, number)
    return out[0] if out else default


def _varint(fields, number, default=0):
    out = fields.get(number, [])
    return out[0][1] if out else default


def _parse_tensor(fields):
    dims = [v for w, v in fields.get(1, []) if w == WIRE_VARINT]
    dtype = _varint(fields, 2, 1)
    name = _string(fields, 8, "")
    raw = fields.get(9, [])
    raw = raw[0][1] if raw else None  # raw_data: WIRE_LEN bytes
    np_dtype = _TENSOR_DTYPES.get(dtype)
    if raw is not None and np_dtype:
        arr = np.frombuffer(raw, dtype=np_dtype).astype(np.float64)
    elif np_dtype:
        # packed numeric fields
        values = []
        if dtype in (1, 11):  # float / double: fixed32/fixed64
            for w, v in fields.get(4, []):
                if w == WIRE_LEN:
                    values.extend(struct.unpack("<%df" % (len(v) // 4), v))
                elif w == WIRE_FIXED32:
                    values.append(struct.unpack("<f", v)[0])
        elif dtype in (6, 7):  # int32/int64: varints
            values = [v for w, v in fields.get(5, []) if w == WIRE_VARINT]
        arr = np.asarray(values, dtype=np.float64)
    else:
        arr = np.zeros(tuple(dims) if dims else (1,))
    if dims:
        arr = arr.reshape(tuple(dims))
    return name, arr


def parse_onnx(data):
    """Parse an ONNX ModelProto. Returns {"graph": {...}}."""
    root = _parse_fields(data, 0, len(data))
    graph_fields_list = [v for w, v in root.get(7, []) if w == WIRE_LEN]
    if not graph_fields_list:
        raise OnnxError("no graph in model")
    graph = _parse_fields(graph_fields_list[0], 0, len(graph_fields_list[0]))

    nodes = []
    for w, v in graph.get(1, []):
        if w != WIRE_LEN:
            continue
        node = _parse_fields(v, 0, len(v))
        attrs = {}
        for aw, av in node.get(5, []):
            if aw != WIRE_LEN:
                continue
            a = _parse_fields(av, 0, len(av))
            name = _string(a, 1, "")
            a_type = _varint(a, 20, 0)
            if a_type == 2:
                attrs[name] = _varint(a, 3, 0)
            elif a_type == 7:
                attrs[name] = _string(a, 7, "")
            elif a_type == 1:
                attrs[name] = struct.unpack("<f", a.get(5, [(0, b"\x00" * 4)])[0][1][:4])[0]
            elif a_type == 9:  # tensor
                t_fields_list = [tv for tw, tv in a.get(9, []) if tw == WIRE_LEN]
                if t_fields_list:
                    t = _parse_fields(t_fields_list[0], 0, len(t_fields_list[0]))
                    _name, arr = _parse_tensor(t)
                    attrs[name] = arr
        nodes.append({
            "inputs": _strings(node, 1),
            "outputs": _strings(node, 2),
            "op": _string(node, 4, ""),
            "attrs": attrs,
        })

    initializers = {}
    for w, v in graph.get(5, []):
        if w != WIRE_LEN:
            continue
        t = _parse_fields(v, 0, len(v))
        name, arr = _parse_tensor(t)
        initializers[name] = arr

    return {
        "nodes": nodes,
        "initializers": initializers,
        "inputs": [i for w, v in graph.get(11, []) if w == WIRE_LEN
                   for i in _strings(_parse_fields(v, 0, len(v)), 1)],
        "outputs": [o for w, v in graph.get(12, []) if w == WIRE_LEN
                    for o in _strings(_parse_fields(v, 0, len(v)), 1)],
    }


def _apply(node, tensors):
    op = node["op"]
    ins = [tensors.get(n) for n in node["inputs"] if n]
    outs = node["outputs"]
    a = node["attrs"]

    def _u(x):
        return np.asarray(x, dtype=np.float64)

    if op == "Constant":
        value = a.get("value")
        result = _u(value) if value is not None else np.zeros(1)
    elif op == "Identity":
        result = _u(ins[0])
    elif op == "Gemm":
        x, w = _u(ins[0]), _u(ins[1])
        b = _u(ins[2]) if len(ins) > 2 and ins[2] is not None else np.zeros(w.shape[0] if w.ndim == 2 else w.shape[-1])
        if a.get("transA", 0):
            x = x.T
        if a.get("transB", 0):
            w = w.T
        result = a.get("alpha", 1.0) * (x @ w) + a.get("beta", 1.0) * b
    elif op == "MatMul":
        result = _u(ins[0]) @ _u(ins[1])
    elif op == "Add":
        result = _u(ins[0]) + _u(ins[1])
    elif op == "Sub":
        result = _u(ins[0]) - _u(ins[1])
    elif op == "Mul":
        result = _u(ins[0]) * _u(ins[1])
    elif op == "Div":
        result = _u(ins[0]) / _u(ins[1])
    elif op == "Relu":
        result = np.maximum(_u(ins[0]), 0.0)
    elif op == "Elu":
        result = np.where(ins[0] > 0.0, _u(ins[0]),
                          a.get("alpha", 1.0) * (np.exp(_u(ins[0])) - 1.0))
    elif op == "Sigmoid":
        result = 1.0 / (1.0 + np.exp(-_u(ins[0])))
    elif op == "Tanh":
        result = np.tanh(_u(ins[0]))
    elif op == "LeakyRelu":
        result = np.where(ins[0] > 0.0, _u(ins[0]), a.get("alpha", 0.01) * _u(ins[0]))
    elif op == "Clip":
        x = _u(ins[0])
        if len(ins) > 1 and ins[1] is not None:
            x = np.maximum(x, float(_u(ins[1])))
        if len(ins) > 2 and ins[2] is not None:
            x = np.minimum(x, float(_u(ins[2])))
        result = x
    elif op == "Reshape":
        x = _u(ins[0])
        shape = ins[1] if len(ins) > 1 and ins[1] is not None else a.get("shape")
        shape = [int(s) for s in np.asarray(shape).ravel()]
        result = x.reshape(shape)
    elif op == "Flatten":
        x = _u(ins[0])
        axis = int(a.get("axis", 1))
        if axis < 0:
            axis = x.ndim + axis
        result = x.reshape((int(np.prod(x.shape[:axis])), int(np.prod(x.shape[axis:]))))
    elif op == "Transpose":
        result = np.transpose(_u(ins[0]), tuple(a.get("perm", None)))
    elif op == "Concat":
        result = np.concatenate([_u(t) for t in ins if t is not None],
                                axis=int(a.get("axis", 0)))
    elif op == "Unsqueeze":
        axes = sorted(int(v) for v in np.asarray(a.get("axes", [0])).ravel())
        result = _u(ins[0])
        for ax in axes:
            result = np.expand_dims(result, axis=ax)
    elif op == "Squeeze":
        axes = [int(v) for v in np.asarray(a.get("axes", [0])).ravel()]
        result = np.squeeze(_u(ins[0]), axis=tuple(axes) if axes else None)
    else:
        raise OnnxError("unsupported ONNX operator %r" % op)

    if not outs:
        return None
    tensors[outs[0]] = result
    return result


def run_onnx(graph, x, clamp_outputs=(0.0, 1.0)):
    """Execute the graph. x: (F,) input vector. Returns output array."""
    tensors = dict(graph["initializers"])
    if graph["inputs"]:
        tensors[graph["inputs"][0]] = np.asarray(x, dtype=np.float64)
    result = None
    for node in graph["nodes"]:
        result = _apply(node, tensors)
    if result is None:
        raise OnnxError("graph produced no output")
    out = np.asarray(result, dtype=np.float64).ravel()
    if clamp_outputs is not None:
        out = np.clip(out, clamp_outputs[0], clamp_outputs[1])
    return out
