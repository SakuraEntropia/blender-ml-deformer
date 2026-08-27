# Copyright (c) 2026 PoseDeformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Unit tests for the bpy-free core. Run with plain Python:

    python3 -m pytest posedeformer/tests/test_core.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from core.features import BoneFeature, FeatureSpec
from core.regressor import LinearRegressor
from core.network import NeuralMorphRegressor
from core import format as fmt


def make_spec():
    spec = FeatureSpec()
    spec.bones = [BoneFeature("bone_a"),
                  BoneFeature("bone_b", use_translation=True, use_scale=True)]
    spec.curve_names = ["c0", "c1"]
    return spec


def test_zero_input_is_all_zeros():
    spec = make_spec()
    B = len(spec.bones)
    x = spec.build_vector(np.zeros((B, 3)), np.zeros((B, 3)), np.zeros((B, 3)),
                          np.zeros(len(spec.curve_names)))
    assert np.allclose(x, 0.0)


def test_feature_layout():
    spec = make_spec()
    # 3 (rot) + 3 (rot) + 3 (trans) + 3 (scale) + 2 (curves) = 14
    assert spec.num_features == 14
    B = len(spec.bones)
    curves = np.array([0.5, 0.25])
    x = spec.build_vector(np.zeros((B, 3)), np.zeros((B, 3)), np.zeros((B, 3)),
                          curves)
    assert x.shape == (14,)
    assert np.allclose(x[:12], 0.0)
    assert np.allclose(x[12:], [0.5, 0.25])


def test_sample_pose_ranges():
    spec = FeatureSpec()
    spec.bones = [BoneFeature("a", rotation_axes=(True, False, False))]
    rng = np.random.default_rng(42)
    rotations, translations, scales = spec.sample_pose(rng, {"a": (0.5, 0.0, 0.0)})
    assert rotations.shape == (1, 3)
    assert abs(rotations[0, 0]) > 0.01
    assert np.allclose(rotations[0, 1:3], 0.0)
    assert np.allclose(translations, 0.0)
    assert np.allclose(scales, 0.0)
    rotations, _, _ = spec.sample_pose(rng, {"a": (0.0, 0.0, 0.0)})
    assert np.allclose(rotations, 0.0)


def test_linear_regressor_recovers_ground_truth():
    rng = np.random.default_rng(1)
    F, N, V = 8, 500, 30
    spec = make_spec()
    X = rng.normal(size=(spec.num_features, N))
    Wt = rng.normal(size=(F, V * 3))
    D = Wt.T @ X[:F] + 1e-3 * rng.normal(size=(V * 3, N))
    model = LinearRegressor()
    model.fit(X, D, regularization=1e-6)
    rel = np.linalg.norm(model.predict(X) - D) / np.linalg.norm(D)
    assert rel < 0.01, "relative error %.4f" % rel


def test_neural_regressor_learns_weights():
    rng = np.random.default_rng(2)
    V, M, F = 20, 4, 16
    N = 600
    morph_deltas = rng.normal(size=(V * 3, M))
    X = rng.normal(size=(F, N))
    y = np.clip(0.4 * X[:M] + 0.2 * X[M:2 * M] + 0.5, 0.0, 1.0)
    model = NeuralMorphRegressor(F, morph_deltas, hidden_sizes=[32, 16],
                                 activation="relu", rng=np.random.default_rng(7))
    loss = model.fit(X, y, iterations=3000, batch_size=64, learning_rate=1e-2,
                     regularization=0.0, seed=0)
    pred = model.predict(X)
    target = morph_deltas @ y
    rel = np.linalg.norm(pred - target) / np.linalg.norm(target)
    assert loss < 1e-3, "final loss %.5f" % loss
    assert rel < 0.02, "relative delta error %.4f" % rel


def test_elu_activation_matches_reference():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    model = NeuralMorphRegressor(4, np.zeros((3, 1)), hidden_sizes=[2],
                                 activation="elu")
    # forward through a 1-layer elu net: out = elu(W x + b)
    model.net.weights[0] = np.eye(1, 4) * 0  # not used directly; test helper below
    from core.network import _ACTIVATIONS
    ref = np.where(x > 0, x, np.exp(x) - 1.0)
    assert np.allclose(_ACTIVATIONS["elu"](x), ref)


def test_format_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    spec = make_spec()
    arrays = {"matrix": rng.normal(size=(spec.num_features, 60)),
              "extra": rng.normal(size=(4,))}
    stats = {"num_features": spec.num_features, "training_loss": 0.00123,
             "num_training_frames": 42}
    path = fmt.save_model(str(tmp_path), "test", "linear", spec, arrays,
                          ["m1", "m2"], stats)
    data = fmt.load_model(str(tmp_path))
    assert data["model_kind"] == "linear"
    assert data["name"] == "test"
    assert data["morph_names"] == ["m1", "m2"]
    assert data["stats"]["training_loss"] == pytest.approx(0.00123)
    assert np.allclose(data["arrays"]["matrix"], arrays["matrix"])
    assert np.allclose(data["arrays"]["extra"], arrays["extra"])
    assert data["spec"].num_features == spec.num_features
    assert [b.name for b in data["spec"].bones] == ["bone_a", "bone_b"]
    assert data["spec"].curve_names == ["c0", "c1"]
    assert Path(path).name == "pose_model.json"


def test_spec_roundtrip_keeps_axes_and_weights():
    spec = make_spec()
    spec.bones[1].rotation_axes = (True, False, True)
    spec.rotation_weight = 2.5
    spec.curve_weight = 0.5
    s2 = FeatureSpec.from_dict(spec.to_dict())
    assert s2.bones[1].rotation_axes == (True, False, True)
    assert s2.rotation_weight == 2.5
    assert s2.curve_weight == 0.5
    assert s2.num_features == spec.num_features
