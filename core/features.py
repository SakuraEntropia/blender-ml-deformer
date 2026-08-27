# Copyright (c) 2026 PoseDeformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Pure-numpy feature specification. No Blender imports.

A pose feature vector is built from per-bone local-space rotation deltas
(axis-angle vectors, radians, relative to the parent's bind frame) followed
by curve values (shape keys in Blender). The rest pose maps to a zero vector,
so a bias-free linear regressor predicts exactly zero deltas at rest.
"""

from __future__ import annotations

import numpy as np


class BoneFeature:
    """One bone's contribution to the feature vector."""

    __slots__ = ("name", "use_rotation", "use_translation", "use_scale",
                 "rotation_axes", "translation_axes", "scale_axes")

    def __init__(self, name, use_rotation=True, use_translation=False,
                 use_scale=False, rotation_axes=(True, True, True),
                 translation_axes=(True, True, True),
                 scale_axes=(True, True, True)):
        self.name = name
        self.use_rotation = use_rotation
        self.use_translation = use_translation
        self.use_scale = use_scale
        self.rotation_axes = tuple(rotation_axes)
        self.translation_axes = tuple(translation_axes)
        self.scale_axes = tuple(scale_axes)

    def num_components(self):
        return ((3 if self.use_rotation else 0)
                + (3 if self.use_translation else 0)
                + (3 if self.use_scale else 0))

    def to_dict(self):
        return {"name": self.name, "use_rotation": self.use_rotation,
                "use_translation": self.use_translation, "use_scale": self.use_scale,
                "rotation_axes": list(self.rotation_axes),
                "translation_axes": list(self.translation_axes),
                "scale_axes": list(self.scale_axes)}

    @staticmethod
    def from_dict(d):
        return BoneFeature(
            d["name"], d.get("use_rotation", True), d.get("use_translation", False),
            d.get("use_scale", False), tuple(d.get("rotation_axes", (True, True, True))),
            tuple(d.get("translation_axes", (True, True, True))),
            tuple(d.get("scale_axes", (True, True, True))))


class FeatureSpec:
    """Bone + curve feature layout with per-component weights."""

    def __init__(self):
        self.bones = []            # list[BoneFeature]
        self.curve_names = []      # shape keys used as curve inputs
        self.rotation_weight = 1.0
        self.translation_weight = 1.0
        self.scale_weight = 1.0
        self.curve_weight = 1.0

    @property
    def num_features(self):
        n = sum(b.num_components() for b in self.bones)
        return n + len(self.curve_names)

    @property
    def bone_names(self):
        return [b.name for b in self.bones]

    def build_vector(self, bone_rotations, bone_translations, bone_scales,
                     curve_values):
        """Compose the (F,) feature vector.

        bone_rotations: (B, 3) axis-angle deltas; zero at bind pose.
        bone_translations: (B, 3); bone_scales: (B, 3) scale - 1.
        curve_values: (C,) raw values.
        """
        bone_rotations = np.asarray(bone_rotations, dtype=np.float64)
        bone_translations = np.asarray(bone_translations, dtype=np.float64)
        bone_scales = np.asarray(bone_scales, dtype=np.float64)
        curve_values = np.asarray(curve_values, dtype=np.float64).ravel()

        parts = []
        for i, b in enumerate(self.bones):
            if b.use_rotation:
                r = np.array([bone_rotations[i][a] if b.rotation_axes[a] else 0.0
                              for a in range(3)])
                parts.append(r * self.rotation_weight)
            if b.use_translation:
                t = np.array([bone_translations[i][a] if b.translation_axes[a] else 0.0
                              for a in range(3)])
                parts.append(t * self.translation_weight)
            if b.use_scale:
                s = np.array([bone_scales[i][a] if b.scale_axes[a] else 0.0
                              for a in range(3)])
                parts.append(s * self.scale_weight)
        out = np.concatenate(parts) if parts else np.empty(0)
        if len(self.curve_names):
            out = np.concatenate([out, curve_values * self.curve_weight])
        return out

    def sample_pose(self, rng, rotation_ranges, translation_ranges=(0.0, 0.0, 0.0),
                    scale_ranges=(0.0, 0.0, 0.0)):
        """Sample a random pose. Ranges are dicts bone name -> (x, y, z)
        half-ranges in RADIANS (rotation) / units. Returns (B,3) arrays."""
        def _ranges(ranges, bone):
            return ranges.get(bone, (0.0, 0.0, 0.0)) if isinstance(ranges, dict) else ranges

        B = len(self.bones)
        rotations = np.zeros((B, 3))
        translations = np.zeros((B, 3))
        scales = np.zeros((B, 3))
        for i, b in enumerate(self.bones):
            rr = np.asarray(_ranges(rotation_ranges, b.name), dtype=np.float64)
            tr = np.asarray(_ranges(translation_ranges, b.name), dtype=np.float64)
            sr = np.asarray(_ranges(scale_ranges, b.name), dtype=np.float64)
            if b.use_rotation:
                e = rng.uniform(-rr, rr)
                rotations[i] = [e[a] if b.rotation_axes[a] else 0.0 for a in range(3)]
            if b.use_translation:
                translations[i] = [rng.uniform(-tr[a], tr[a]) if b.translation_axes[a] else 0.0
                                   for a in range(3)]
            if b.use_scale:
                scales[i] = [rng.uniform(-sr[a], sr[a]) if b.scale_axes[a] else 0.0
                             for a in range(3)]
        return rotations, translations, scales

    def to_dict(self):
        return {
            "bones": [b.to_dict() for b in self.bones],
            "curve_names": list(self.curve_names),
            "rotation_weight": self.rotation_weight,
            "translation_weight": self.translation_weight,
            "scale_weight": self.scale_weight,
            "curve_weight": self.curve_weight,
        }

    @staticmethod
    def from_dict(d):
        spec = FeatureSpec()
        spec.bones = [BoneFeature.from_dict(b) for b in d.get("bones", [])]
        spec.curve_names = list(d.get("curve_names", []))
        spec.rotation_weight = d.get("rotation_weight", 1.0)
        spec.translation_weight = d.get("translation_weight", 1.0)
        spec.scale_weight = d.get("scale_weight", 1.0)
        spec.curve_weight = d.get("curve_weight", 1.0)
        return spec
