# Copyright (c) 2026 PoseDeformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Property groups for the PoseDeformer workspace (scene.psd)."""

import bpy


def _poll_armature(self, obj):
    return obj is not None and obj.type == "ARMATURE"


def _poll_mesh(self, obj):
    return obj is not None and obj.type == "MESH"


class PSD_BoneEntry(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Bone")
    use_rotation: bpy.props.BoolProperty(name="Use Rotation", default=True)
    use_rotation_x: bpy.props.BoolProperty(name="X", default=True)
    use_rotation_y: bpy.props.BoolProperty(name="Y", default=True)
    use_rotation_z: bpy.props.BoolProperty(name="Z", default=True)
    rotation_range_x: bpy.props.FloatProperty(
        name="RX Range", description="Training rotation half-range (radians)",
        default=0.2618, min=0.0, max=3.1416, subtype="ANGLE")
    rotation_range_y: bpy.props.FloatProperty(
        name="RY Range", default=0.2618, min=0.0, max=3.1416, subtype="ANGLE")
    rotation_range_z: bpy.props.FloatProperty(
        name="RZ Range", default=0.2618, min=0.0, max=3.1416, subtype="ANGLE")
    use_translation: bpy.props.BoolProperty(name="Use Translation", default=False)
    use_scale: bpy.props.BoolProperty(name="Use Scale", default=False)


class PSD_KeyEntry(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Shape Key")
    use: bpy.props.BoolProperty(name="Use", default=False)


class PSD_TrainingClip(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Action")
    use: bpy.props.BoolProperty(name="Use", default=True)
    frame_start: bpy.props.IntProperty(name="Start", default=1, min=0)
    frame_end: bpy.props.IntProperty(name="End", default=100, min=0)
    num_frames: bpy.props.IntProperty(name="Frames", default=20, min=1, max=10000)


class PSD_Settings(bpy.types.PropertyGroup):
    # ---- Setup ----
    armature: bpy.props.PointerProperty(name="Armature", type=bpy.types.Object,
                                        poll=_poll_armature)
    mesh: bpy.props.PointerProperty(name="Mesh", type=bpy.types.Object,
                                    poll=_poll_mesh)
    model_kind: bpy.props.EnumProperty(
        name="Model",
        items=(
            ("LINEAR", "Linear Model",
             "Linear pose-to-delta regressor (closed-form ridge fit)"),
            ("NEURAL", "Neural Model",
             "Feed-forward network predicting morph target weights"),
        ),
        default="LINEAR",
    )

    # ---- Inputs ----
    bones: bpy.props.CollectionProperty(type=PSD_BoneEntry)
    active_bone_index: bpy.props.IntProperty()
    curve_inputs: bpy.props.CollectionProperty(type=PSD_KeyEntry)
    active_curve_index: bpy.props.IntProperty()
    morph_targets: bpy.props.CollectionProperty(type=PSD_KeyEntry)
    active_morph_index: bpy.props.IntProperty()
    input_rotation_weight: bpy.props.FloatProperty(
        name="Rotation Weight", default=1.0, min=0.0, max=100.0)
    input_translation_weight: bpy.props.FloatProperty(
        name="Translation Weight", default=1.0, min=0.0, max=100.0)
    input_scale_weight: bpy.props.FloatProperty(
        name="Scale Weight", default=1.0, min=0.0, max=100.0)
    input_curve_weight: bpy.props.FloatProperty(
        name="Curve Weight", default=1.0, min=0.0, max=100.0)

    # ---- Training ----
    num_random_poses: bpy.props.IntProperty(name="Random Poses", default=100,
                                            min=1, max=10000)
    random_seed: bpy.props.IntProperty(name="Seed", default=0, min=0,
                                       description="0 = random every time")
    use_clip_sampling: bpy.props.BoolProperty(name="Sample From Actions",
                                              default=False)
    training_clips: bpy.props.CollectionProperty(type=PSD_TrainingClip)
    active_clip_index: bpy.props.IntProperty()
    morph_zero_prob: bpy.props.FloatProperty(
        name="Morph Zero Probability", default=0.3, min=0.0, max=1.0,
        description="Chance a morph weight is 0 in a training sample (Neural only)")

    # ---- Model ----
    hidden_layers: bpy.props.StringProperty(
        name="Hidden Layers", default="64, 32",
        description="Comma separated hidden layer sizes (Neural only)")
    learning_rate: bpy.props.FloatProperty(name="Learning Rate", default=1e-3,
                                           min=1e-6, max=1.0)
    iterations: bpy.props.IntProperty(name="Iterations", default=3000,
                                      min=1, max=200000)
    batch_size: bpy.props.IntProperty(name="Batch Size", default=64,
                                      min=1, max=4096)
    regularization: bpy.props.FloatProperty(name="Regularization", default=1e-4,
                                            min=0.0, max=1.0)
    clamp_morph_weights: bpy.props.BoolProperty(
        name="Clamp Morph Weights", default=True,
        description="Clamp predicted morph weights to [0, 1]")

    # ---- Preview / IO ----
    auto_refresh: bpy.props.BoolProperty(name="Auto Refresh Preview", default=True)
    preview_object: bpy.props.PointerProperty(name="Preview", type=bpy.types.Object)
    num_bake_poses: bpy.props.IntProperty(name="Bake Poses", default=20,
                                          min=1, max=500)
    model_dir: bpy.props.StringProperty(name="Model Directory", subtype="DIR_PATH")
    model_name: bpy.props.StringProperty(name="Model Name", default="posedeformer")

    # ---- Engine bridge ----
    engine_clamp_weights: bpy.props.BoolProperty(
        name="Clamp Imported Weights", default=True,
        description="Clamp morph weights to [0, 1] when running imported networks")

    # ---- Stats ----
    is_trained: bpy.props.BoolProperty(name="Trained", default=False)
    num_features: bpy.props.IntProperty(name="Features", default=0)
    num_vertices: bpy.props.IntProperty(name="Vertices", default=0)
    num_training_frames: bpy.props.IntProperty(name="Training Frames", default=0)
    training_loss: bpy.props.FloatProperty(name="Loss", default=0.0)
    max_vertex_error: bpy.props.FloatProperty(name="Max Vertex Error", default=0.0)


classes = (
    PSD_BoneEntry,
    PSD_KeyEntry,
    PSD_TrainingClip,
    PSD_Settings,
)


def register():
    # property group classes are registered by the add-on __init__ loop
    bpy.types.Scene.psd = bpy.props.PointerProperty(type=PSD_Settings)


def unregister():
    del bpy.types.Scene.psd
