# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Sidebar UI (3D viewport, N key, "Pose Deformer" tab)."""

import bpy


class BMD_UL_bones(bpy.types.UIList):
    bl_idname = "BMD_UL_bones"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.prop(item, "use_rotation", text="", emboss=False)
            layout.label(text=item.name, icon="BONE_DATA")


class BMD_UL_keys(bpy.types.UIList):
    bl_idname = "BMD_UL_keys"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.prop(item, "use", text="", emboss=False)
            layout.label(text=item.name, icon="SHAPEKEY_DATA")


class BMD_UL_clips(bpy.types.UIList):
    bl_idname = "BMD_UL_clips"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.prop(item, "use", text="", emboss=False)
            layout.label(text=item.name, icon="ACTION")


class BMD_PT_main(bpy.types.Panel):
    bl_label = "Pose Deformer"
    bl_idname = "BMD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pose Deformer"

    def draw(self, context):
        pass


class BMD_PT_setup(bpy.types.Panel):
    bl_label = "Setup"
    bl_parent_id = "BMD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.bmd
        col = layout.column(align=True)
        col.prop(settings, "armature")
        col.prop(settings, "mesh")
        layout.separator()
        layout.prop(settings, "model_kind")


class BMD_PT_inputs(bpy.types.Panel):
    bl_label = "Inputs"
    bl_parent_id = "BMD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bmd

        box = layout.box()
        box.label(text="Bones", icon="BONE_DATA")
        box.template_list("BMD_UL_bones", "", settings, "bones",
                          settings, "active_bone_index", rows=4)
        row = box.row(align=True)
        row.operator("bmd.sync_bones", icon="IMPORT")
        row.operator("bmd.bones_remove", icon="X")
        idx = settings.active_bone_index
        if 0 <= idx < len(settings.bones):
            bone = settings.bones[idx]
            sub = box.box()
            sub.label(text=bone.name, icon="BONE_DATA")
            sub.prop(bone, "use_rotation")
            if bone.use_rotation:
                crow = sub.row(align=True)
                crow.prop(bone, "use_rotation_x", text="X", toggle=1)
                crow.prop(bone, "use_rotation_y", text="Y", toggle=1)
                crow.prop(bone, "use_rotation_z", text="Z", toggle=1)
                rrow = sub.row(align=True)
                rrow.prop(bone, "rotation_range_x", text="Range")
                rrow.prop(bone, "rotation_range_y", text="")
                rrow.prop(bone, "rotation_range_z", text="")
            sub.prop(bone, "use_translation")
            sub.prop(bone, "use_scale")

        box = layout.box()
        box.label(text="Curve Inputs (Shape Keys)", icon="SHAPEKEY_DATA")
        box.template_list("BMD_UL_keys", "", settings, "curve_inputs",
                          settings, "active_curve_index", rows=3)
        row = box.row(align=True)
        row.operator("bmd.sync_curve_inputs", icon="IMPORT")
        row.operator("bmd.key_remove", icon="X").target = "CURVE"

        box = layout.box()
        box.label(text="Morph Targets (Neural)", icon="SHAPEKEY_DATA")
        box.template_list("BMD_UL_keys", "", settings, "morph_targets",
                          settings, "active_morph_index", rows=3)
        row = box.row(align=True)
        row.operator("bmd.sync_morph_targets", icon="IMPORT")
        row.operator("bmd.key_remove", icon="X").target = "MORPH"


class BMD_PT_training(bpy.types.Panel):
    bl_label = "Training"
    bl_parent_id = "BMD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.bmd
        col = layout.column(align=True)
        col.prop(settings, "num_random_poses")
        col.prop(settings, "random_seed")
        layout.prop(settings, "bake_poses_to_timeline")
        if settings.bake_poses_to_timeline:
            layout.prop(settings, "timeline_start_frame")
        layout.prop(settings, "use_clip_sampling")
        if settings.use_clip_sampling:
            box = layout.box()
            box.template_list("BMD_UL_clips", "", settings, "training_clips",
                              settings, "active_clip_index", rows=3)
            row = box.row(align=True)
            row.operator("bmd.clip_add", icon="ADD", text="Add")
            row.operator("bmd.clip_add_all", icon="IMPORT")
            row.operator("bmd.clip_remove", icon="X")
            idx = settings.active_clip_index
            if 0 <= idx < len(settings.training_clips):
                clip = settings.training_clips[idx]
                sub = box.box()
                sub.prop(clip, "name", text="Action")
                sub.prop(clip, "frame_start")
                sub.prop(clip, "frame_end")
                sub.prop(clip, "num_frames")
        if settings.model_kind == "NEURAL":
            layout.prop(settings, "morph_zero_prob")
        layout.separator()
        layout.operator("bmd.generate_training_data", icon="PLAY")


class BMD_PT_model(bpy.types.Panel):
    bl_label = "Model"
    bl_parent_id = "BMD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.bmd

        box = layout.box()
        box.label(text="Input Weights")
        box.prop(settings, "input_rotation_weight")
        box.prop(settings, "input_translation_weight")
        box.prop(settings, "input_scale_weight")
        box.prop(settings, "input_curve_weight")

        if settings.model_kind == "LINEAR":
            layout.prop(settings, "regularization")
        else:
            layout.prop(settings, "hidden_layers")
            col = layout.column(align=True)
            col.prop(settings, "learning_rate")
            col.prop(settings, "iterations")
            col.prop(settings, "batch_size")
            layout.prop(settings, "regularization")
            layout.prop(settings, "clamp_morph_weights")

        layout.separator()
        layout.operator("bmd.train", icon="PLAY")

        if settings.is_trained:
            box = layout.box()
            box.label(text="Stats", icon="INFO")
            col = box.column(align=True)
            col.label(text="Features: %d" % settings.num_features)
            col.label(text="Vertices: %d" % settings.num_vertices)
            col.label(text="Training Frames: %d" % settings.num_training_frames)
            col.label(text="Loss: %.6f" % settings.training_loss)
            col.label(text="Max Vertex Error: %.4f" % settings.max_vertex_error)


class BMD_PT_preview(bpy.types.Panel):
    bl_label = "Preview & IO"
    bl_parent_id = "BMD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.bmd

        layout.prop(settings, "auto_refresh")
        col = layout.column(align=True)
        col.operator("bmd.create_preview_proxy", icon="MESH_DATA")
        col.operator("bmd.refresh_preview", icon="FILE_REFRESH")

        layout.separator()
        col = layout.column(align=True)
        col.prop(settings, "num_bake_poses")
        col.operator("bmd.bake_shape_keys", icon="SHAPEKEY_DATA")

        layout.separator()
        box = layout.box()
        box.label(text="Model Files")
        box.prop(settings, "model_dir")
        box.prop(settings, "model_name")
        row = box.row(align=True)
        row.operator("bmd.export_model", icon="EXPORT")
        row.operator("bmd.import_model", icon="IMPORT")

        layout.separator()
        layout.operator("bmd.clear", icon="TRASH")


class BMD_PT_engine(bpy.types.Panel):
    bl_label = "Engine Bridge"
    bl_parent_id = "BMD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.bmd

        box = layout.box()
        box.label(text="Import", icon="IMPORT")
        box.operator("bmd.import_engine_nmn")
        box.operator("bmd.import_engine_onnx")
        box.prop(settings, "engine_clamp_weights")
        box.label(text=("Imported networks map: bones -> first N bone "
                        "entries, curves/morphs -> first N enabled keys"),
                  icon="INFO")

        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        box.operator("bmd.export_engine_nmn")
        box.label(text=("Exports the trained Neural model as a global-mode "
                        ".nmn network (retrains on the engine input layout)"),
                  icon="INFO")


classes = (
    BMD_UL_bones,
    BMD_UL_keys,
    BMD_UL_clips,
    BMD_PT_main,
    BMD_PT_setup,
    BMD_PT_inputs,
    BMD_PT_training,
    BMD_PT_model,
    BMD_PT_preview,
    BMD_PT_engine,
)
