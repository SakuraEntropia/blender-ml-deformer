# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

"""Operators. Long work runs through modal operators driven by the generator
functions in ``train`` / ``ue``; ESC cancels and restores the scene state."""

import os

import bpy

from . import bridge
from . import train as train_ops
from . import ue as ue_ops


# ---------------------------------------------------------------------------
# List management
# ---------------------------------------------------------------------------

class BMD_OT_sync_bones(bpy.types.Operator):
    bl_idname = "bmd.sync_bones"
    bl_label = "Load Bones From Armature"
    bl_description = ("Rebuild the bone input list from the armature "
                     "(existing per-bone settings are kept by name)")

    def execute(self, context):
        n = bridge.sync_bones(context.scene.bmd)
        self.report({"INFO"}, "Loaded %d bones" % n)
        return {"FINISHED"}


class BMD_OT_sync_curve_inputs(bpy.types.Operator):
    bl_idname = "bmd.sync_curve_inputs"
    bl_label = "Load Shape Keys"
    bl_description = "List the mesh shape keys as curve inputs"

    def execute(self, context):
        n = bridge.sync_curve_inputs(context.scene.bmd)
        self.report({"INFO"}, "Loaded %d shape keys" % n)
        return {"FINISHED"}


class BMD_OT_sync_morph_targets(bpy.types.Operator):
    bl_idname = "bmd.sync_morph_targets"
    bl_label = "Load Shape Keys"
    bl_description = "List the mesh shape keys as morph targets (Neural model)"

    def execute(self, context):
        n = bridge.sync_morph_targets(context.scene.bmd)
        self.report({"INFO"}, "Loaded %d shape keys" % n)
        return {"FINISHED"}


class BMD_OT_bones_remove(bpy.types.Operator):
    bl_idname = "bmd.bones_remove"
    bl_label = "Remove Bone"
    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        settings = context.scene.bmd
        i = self.index if self.index >= 0 else settings.active_bone_index
        if 0 <= i < len(settings.bones):
            settings.bones.remove(i)
            settings.active_bone_index = min(i, len(settings.bones) - 1)
        return {"FINISHED"}


class BMD_OT_key_remove(bpy.types.Operator):
    bl_idname = "bmd.key_remove"
    bl_label = "Remove Key"
    target: bpy.props.StringProperty(default="CURVE")  # CURVE | MORPH
    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        settings = context.scene.bmd
        if self.target == "MORPH":
            coll, active = settings.morph_targets, "active_morph_index"
        else:
            coll, active = settings.curve_inputs, "active_curve_index"
        i = self.index if self.index >= 0 else getattr(settings, active)
        if 0 <= i < len(coll):
            coll.remove(i)
            setattr(settings, active, min(i, len(coll) - 1))
        return {"FINISHED"}


class BMD_OT_clip_add(bpy.types.Operator):
    bl_idname = "bmd.clip_add"
    bl_label = "Add Action Slot"
    action: bpy.props.StringProperty(name="Action Name")

    def execute(self, context):
        settings = context.scene.bmd
        e = settings.training_clips.add()
        e.name = self.action
        settings.active_clip_index = len(settings.training_clips) - 1
        return {"FINISHED"}


class BMD_OT_clip_add_all(bpy.types.Operator):
    bl_idname = "bmd.clip_add_all"
    bl_label = "Add All Actions"

    def execute(self, context):
        settings = context.scene.bmd
        existing = {e.name for e in settings.training_clips}
        for act in bpy.data.actions:
            if act.name not in existing:
                e = settings.training_clips.add()
                e.name = act.name
        self.report({"INFO"}, "%d actions listed" % len(settings.training_clips))
        return {"FINISHED"}


class BMD_OT_clip_remove(bpy.types.Operator):
    bl_idname = "bmd.clip_remove"
    bl_label = "Remove Action Slot"
    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        settings = context.scene.bmd
        i = self.index if self.index >= 0 else settings.active_clip_index
        if 0 <= i < len(settings.training_clips):
            settings.training_clips.remove(i)
            settings.active_clip_index = min(i, len(settings.training_clips) - 1)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Modal base
# ---------------------------------------------------------------------------

class BMD_ProgressOperator(bpy.types.Operator):
    """Runs a generator in a modal loop with a progress bar; ESC cancels."""

    bl_options = {"REGISTER"}
    _timer = None
    _gen = None
    _cancelled = False

    def _make_generator(self, context):
        raise NotImplementedError

    def _on_done(self, context):
        pass

    def invoke(self, context, event):
        try:
            self._gen = self._make_generator(context)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        wm.progress_begin(0, 100)
        return {"RUNNING_MODAL"}

    def _abort(self, context, message=None):
        if self._gen is not None:
            try:
                self._gen.close()  # runs the generator's finally: restores scene
            except Exception:
                pass
            self._gen = None
        self._finish(context, cancelled=True, message=message)

    def _finish(self, context, cancelled, message=None):
        wm = context.window_manager
        wm.progress_end()
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if cancelled:
            self.report({"WARNING"}, message or "Cancelled")
        else:
            try:
                self._on_done(context)
            except Exception as exc:
                self.report({"ERROR"}, str(exc))
            self.report({"INFO"}, "Done")

    def modal(self, context, event):
        wm = context.window_manager
        if event.type == "ESC":
            self._cancelled = True
            return {"RUNNING_MODAL"}
        if event.type != "TIMER" or self._gen is None:
            return {"PASS_THROUGH"}
        if self._cancelled:
            self._abort(context, "Cancelled")
            return {"FINISHED"}
        try:
            frac = next(self._gen)
            wm.progress_update(int(frac * 100))
            if frac >= 1.0:
                try:
                    next(self._gen)
                except StopIteration:
                    pass
                self._gen = None
                self._finish(context, cancelled=False)
                return {"FINISHED"}
        except StopIteration:
            self._gen = None
            self._finish(context, cancelled=False)
            return {"FINISHED"}
        except Exception as exc:
            self._abort(context, str(exc))
            return {"FINISHED"}
        return {"PASS_THROUGH"}


# ---------------------------------------------------------------------------
# Long operations
# ---------------------------------------------------------------------------

class BMD_OT_generate_training_data(BMD_ProgressOperator):
    bl_idname = "bmd.generate_training_data"
    bl_label = "Generate Training Data"

    def _make_generator(self, context):
        return train_ops.generate_training_data_iter(context.scene.bmd)

    def _on_done(self, context):
        self.report({"INFO"}, "Generated %d training frames"
                    % context.scene.bmd.num_training_frames)


class BMD_OT_train(BMD_ProgressOperator):
    bl_idname = "bmd.train"
    bl_label = "Train Model"

    def _make_generator(self, context):
        return train_ops.train_model_iter(context.scene.bmd)


class BMD_OT_bake_shape_keys(BMD_ProgressOperator):
    bl_idname = "bmd.bake_shape_keys"
    bl_label = "Bake Shape Keys"

    def _make_generator(self, context):
        return train_ops.bake_shape_keys_iter(context.scene.bmd)


class BMD_OT_export_engine_nmn(BMD_ProgressOperator):
    bl_idname = "bmd.export_engine_nmn"
    bl_label = "Export Engine Network (.nmn)"
    bl_description = ("Retrain the neural model on the engine input layout and "
                      "write a global-mode .nmn network file")
    filepath: bpy.props.StringProperty(name="File Path", subtype="FILE_PATH")

    def invoke(self, context, event):
        base = context.scene.bmd.model_dir or (os.path.dirname(bpy.data.filepath)
                                               if bpy.data.filepath else "")
        self.filepath = os.path.join(base, (context.scene.bmd.model_name
                                            or "blender_ml_deformer") + ".nmn")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def _make_generator(self, context):
        if not self.filepath:
            raise ValueError("no file selected")
        return ue_ops.export_nmn_iter(context.scene.bmd, self.filepath)

    def _on_done(self, context):
        self.report({"INFO"}, "Wrote %s" % self.filepath)


# ---------------------------------------------------------------------------
# Preview / IO
# ---------------------------------------------------------------------------

class BMD_OT_create_preview_proxy(bpy.types.Operator):
    bl_idname = "bmd.create_preview_proxy"
    bl_label = "Create Preview Proxy"

    def execute(self, context):
        try:
            obj = bridge.create_preview_proxy(context)
            self.report({"INFO"}, "Created preview proxy %r" % obj.name)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BMD_OT_refresh_preview(bpy.types.Operator):
    bl_idname = "bmd.refresh_preview"
    bl_label = "Refresh Preview"

    def execute(self, context):
        bridge.refresh_preview(context.scene, force=True)
        return {"FINISHED"}


class BMD_OT_export_model(bpy.types.Operator):
    bl_idname = "bmd.export_model"
    bl_label = "Export Model"
    filepath: bpy.props.StringProperty(name="File Path", subtype="FILE_PATH")

    def invoke(self, context, event):
        settings = context.scene.bmd
        base = settings.model_dir or (os.path.dirname(bpy.data.filepath)
                                      if bpy.data.filepath else "")
        self.filepath = os.path.join(base, (settings.model_name
                                            or "blender_ml_deformer") + ".json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        settings = context.scene.bmd
        if self.filepath:
            settings.model_dir = os.path.dirname(self.filepath)
        try:
            path = train_ops.export_model(settings)
            self.report({"INFO"}, "Exported to %s" % path)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BMD_OT_import_model(bpy.types.Operator):
    bl_idname = "bmd.import_model"
    bl_label = "Import Model"
    filepath: bpy.props.StringProperty(name="File Path", subtype="FILE_PATH")

    def invoke(self, context, event):
        settings = context.scene.bmd
        base = settings.model_dir or (os.path.dirname(bpy.data.filepath)
                                      if bpy.data.filepath else "")
        self.filepath = os.path.join(base, "pose_model.json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        settings = context.scene.bmd
        if self.filepath:
            settings.model_dir = os.path.dirname(self.filepath)
        try:
            name = train_ops.import_model(settings)
            self.report({"INFO"}, "Imported model %r" % name)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BMD_OT_import_engine_nmn(bpy.types.Operator):
    bl_idname = "bmd.import_engine_nmn"
    bl_label = "Import Engine Network (.nmn)"
    filepath: bpy.props.StringProperty(name="File Path", subtype="FILE_PATH")

    def invoke(self, context, event):
        base = context.scene.bmd.model_dir or (os.path.dirname(bpy.data.filepath)
                                               if bpy.data.filepath else "")
        self.filepath = os.path.join(base, "model.nmn")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            name = ue_ops.import_nmn(context.scene.bmd, self.filepath)
            self.report({"INFO"}, "Imported engine network %r" % name)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BMD_OT_import_engine_onnx(bpy.types.Operator):
    bl_idname = "bmd.import_engine_onnx"
    bl_label = "Import ONNX Network"
    filepath: bpy.props.StringProperty(name="File Path", subtype="FILE_PATH")

    def invoke(self, context, event):
        base = context.scene.bmd.model_dir or (os.path.dirname(bpy.data.filepath)
                                               if bpy.data.filepath else "")
        self.filepath = os.path.join(base, "model.onnx")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            name = ue_ops.import_onnx(context.scene.bmd, self.filepath)
            self.report({"INFO"}, "Imported ONNX network %r" % name)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class BMD_OT_clear(bpy.types.Operator):
    bl_idname = "bmd.clear"
    bl_label = "Clear State"

    def execute(self, context):
        settings = context.scene.bmd
        bridge.clear_runtime()
        settings.is_trained = False
        settings.num_features = 0
        settings.num_vertices = 0
        settings.num_training_frames = 0
        settings.training_loss = 0.0
        settings.max_vertex_error = 0.0
        if settings.preview_object is not None:
            bpy.data.objects.remove(settings.preview_object, do_unlink=True)
            settings.preview_object = None
        for obj in [o for o in context.scene.objects
                    if o.name.endswith(("_BMDPreview", "_BMDBaked"))]:
            bpy.data.objects.remove(obj, do_unlink=True)
        self.report({"INFO"}, "Cleared Blender ML Deformer state")
        return {"FINISHED"}


classes = (
    BMD_OT_sync_bones,
    BMD_OT_sync_curve_inputs,
    BMD_OT_sync_morph_targets,
    BMD_OT_bones_remove,
    BMD_OT_key_remove,
    BMD_OT_clip_add,
    BMD_OT_clip_add_all,
    BMD_OT_clip_remove,
    BMD_OT_generate_training_data,
    BMD_OT_train,
    BMD_OT_bake_shape_keys,
    BMD_OT_export_engine_nmn,
    BMD_OT_create_preview_proxy,
    BMD_OT_refresh_preview,
    BMD_OT_export_model,
    BMD_OT_import_model,
    BMD_OT_import_engine_nmn,
    BMD_OT_import_engine_onnx,
    BMD_OT_clear,
)
