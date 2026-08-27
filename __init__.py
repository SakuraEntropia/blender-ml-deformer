# Copyright (c) 2026 Blender ML Deformer contributors.
# Licensed under the MIT License. See LICENSE in the project root.

bl_info = {
    "name": "Blender ML Deformer",
    "author": "Blender ML Deformer contributors",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar (N) > Pose Deformer",
    "description": "Train pose-driven mesh deformation models inside Blender, "
                   "preview them in the viewport, bake results to shape keys, "
                   "and exchange networks with the engine-format (.nmn / ONNX).",
    "category": "Animation",
    "support": "COMMUNITY",
}

try:
    import bpy
except ModuleNotFoundError:
    # Outside Blender (core unit tests): only the bpy-free `core` package is
    # importable, everything else needs bpy.
    bpy = None
    classes = ()

    def register():
        raise RuntimeError("Blender ML Deformer requires Blender (bpy)")

    def unregister():
        pass
else:
    from . import props
    from . import ops
    from . import ui
    from . import bridge
    from . import train

    classes = (
        *props.classes,
        *ui.classes,
        *ops.classes,
    )

    def register():
        for cls in classes:
            bpy.utils.register_class(cls)
        props.register()
        bridge.register_handlers()

    def unregister():
        bridge.unregister_handlers()
        bridge.clear_runtime()
        props.unregister()
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
