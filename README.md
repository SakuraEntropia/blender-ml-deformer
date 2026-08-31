# Blender ML Deformer

一套**全新独立实现**的 Blender 姿态驱动网格形变工具：在 Blender 里生成训练数据、训练模型、视口预览形变、烘焙 Shape Keys，并可以**交换引擎格式的网络文件**。代码从头编写（MIT 许可），不包含任何第三方引擎源码，只按引擎运行时公开的文件布局做互操作。

- 目标版本：Blender 4.0+（在 Blender 5.2 Alpha 上端到端测试通过）
- 依赖：仅 numpy（Blender 自带）
- 面板：3D 视图 N 键侧边栏 → **Pose Deformer**

## 功能总览

| 区块 | 功能 |
|---|---|
| Setup | 选择骨架/网格，选模型类型（Linear / Neural） |
| Inputs | 骨骼列表（每骨骼三轴开关 + 训练旋转范围）、Shape Key 曲线输入、Morph Target |
| Training | **Generate Random Poses：直接往时间轴生成随机姿态**（rest 参考帧 + N 个随机姿态逐帧 keyframe，不跑网格求值）；随机姿态采样（种子可控）、动作帧采样（Start/End/Frames）、Generate Training Data（同步把训练姿态烘焙到时间轴）、模态进度 + ESC 恢复场景 |
| Model | Linear：岭回归闭式解；Neural：MLP 预测 morph 权重（Adam、输入标准化、权重钳制、可配隐藏层/学习率/迭代数），训练后显示 Stats |
| Preview & IO | 预览代理网格（自动刷新 + 手动）、一键烘焙相对 Shape Keys + 姿态库 JSON、自有格式导出/导入（`pose_model.json` + `pose_model.npz`） |
| **Engine Bridge** | **导入引擎格式网络（`.nmn` / `.onnx`）在 Blender 里跑；把 Blender 训练的 Neural 模型导出成引擎 `.nmn` 网络** |

## 使用流程

1. **Setup**：选 Armature + Mesh，选模型类型。
2. **Inputs**：Load Bones From Armature；需要曲线输入/变形目标时勾选对应 Shape Key。
3. **Training**：设随机姿态数/种子 → Generate Training Data（或开启动作采样）。生成的每个姿态会按顺序 keyframe 到骨架时间轴（Bake Poses To Timeline，默认开启，Start Frame 可调；生成结束时间轴停在起始帧，直接空格播放即可回看训练姿态）。
4. **Model**：调参 → Train。
5. **Preview & IO**：Create Preview Proxy（摆姿态自动跟随）；Bake Shape Keys 输出烘焙对象；Export/Import Model 存自有格式。

## Engine Bridge（引擎格式兼容）

### 导入 `.nmn`（引擎 6 的神经形变网络交换格式）

- 文件不存名字、只存数量，导入按当前列表顺序映射：
  - 骨骼 ← Inputs > Bones 的**前 N 项**（需先 sync）
  - 曲线 ← 已勾选的 Curve Inputs 的**前 N 项**
  - 变形目标 ← 已勾选的 Morph Targets 的**前 N 项**（数量必须一致）
- 导入后即可预览/烘焙，与本地训练的模型走同一套管线。
- 支持 global 模式完整执行；local 模式（带组网络）会给出警告（组网络不执行）。
- 导入时权重钳制 [0,1] 默认开启（`Clamp Imported Weights`）。

### 导入 `.onnx`

- 为引擎 5.4/5.5 时代训练管线导出的 MLP 网络提供兼容（纯 numpy 执行，无需 onnx/onnxruntime）。
- 支持的算子：Gemm / MatMul / Add / Sub / Mul / Div / Relu / Elu / Sigmoid / Tanh / LeakyRelu / Clip / Constant / Identity / Reshape / Flatten / Transpose / Concat / Unsqueeze / Squeeze。
- 输入特征数 = 骨骼数 × 6 + 曲线数（与当前 Inputs 设置核对，不一致会报错）。

### 导出 `.nmn`

- 只对**本插件训练的 Neural 模型**开放。
- 导出时会在同一份训练数据上按引擎输入布局（每骨骼 6 浮点 = 局部旋转矩阵前两列）**重新训练一个 ELU MLP**，然后写出 global 模式 `.nmn`（含输入均值/方差、内嵌字节码模型），并附一个 `*.bmd_ue.json` 侧车文件记录骨骼/曲线/变形目标的名字映射。
- 引擎侧放置位置：项目 Intermediate 目录下的模型子目录（引擎编辑器按 `<Intermediate>/<ClassName>/<ClassName>.nmn` 查找），详见引擎的模型训练网络路径约定。

### 格式事实（互操作依据，均来自引擎运行时的公开行为）

- `.nmn`：magic `0x234A1304`，version 1；头 9 个 uint32（模式/变形数/每骨骼变形数/骨骼数/曲线数/组数/每组项数/每曲线浮点数）；64 字节对齐的输入均值、方差 float 数组；长度前缀运行时名字符串；64 字节对齐的内嵌模型字节块。
- 内嵌模型：magic `0x0BA51C01`，version 1；层级树 Sequence(1)/Linear(4)/ReLU(7)/ELU(8)/TanH(9)/GELU(20)；uint32 按 4 字节对齐，float 数组按 64 字节对齐；Linear 权重按 `[输入][输出]` 行主序存储，每层后跟 ELU。
- 骨骼输入：6 浮点 = 骨骼相对父级的局部旋转四元数对应 3x3 矩阵的第 0、1 列；曲线输入为原始值；推理前做 `(x - mean) / std`。

## 设计要点

- **旋转特征用轴角向量**（每骨骼 3 分量）：rest 姿态对应零向量，无偏置线性模型在 rest 处**结构上保证零输出**；小角度下对线性模型一阶精确。
- 训练数据显式包含一个 rest 参考帧（零输入 → 零偏移）。
- Neural 类模型（含引擎导入的网络）的预览 base 取**当前姿态蒙皮结果**（修正量叠加其上）；Linear 模型 base 取 rest 姿态（含完整形变）——两类语义与引擎一致。
- 所有长任务为模态进度条 + ESC 取消；场景姿态/Shape Key 值在中断时也会恢复（生成器 finally 保证）。

## 限制

- 网格 modifier 不能改拓扑（生成阶段会检测顶点数变化并报错）。
- 随机采样不检测碰撞；穿模场景用动作采样。
- Linear 模型在大角度（>~20°）下精度下降（线性近似天性，引擎同类模型同样如此）。
- 引擎侧变形目标资产（顶点偏移矩阵资产）不在本工具覆盖范围；本工具交换的是**网络**（.nmn / .onnx）。

## 测试

```bash
# 纯数学与格式层单测（系统 python，无需 Blender；含按格式规格手造的二进制参考字节验证）
python3 -m pytest blender_ml_deformer/tests -q

# Blender headless 端到端冒烟（含 .nmn 导出→重新导入→推理一致、ONNX 导入）
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
    --python blender_ml_deformer/tests/blender_smoke.py
```

## 目录结构

```
blender_ml_deformer/
├── __init__.py      # 注册入口（bpy 缺失时仅 core 可导入，便于测试）
├── props.py         # scene.bmd 属性组
├── ops.py           # 操作符（模态进度）
├── ui.py            # N 面板
├── bridge.py        # bpy 侧运行时（姿态/网格求值、预览、烘焙、handlers）
├── train.py         # 训练编排生成器 + 自有格式存取
├── ue.py            # 引擎桥（.nmn/.onnx 导入、.nmn 导出）
├── core/            # 纯 numpy 层（无 bpy）
│   ├── features.py  #   特征规格与姿态采样（轴角旋转）
│   ├── regressor.py #   线性岭回归
│   ├── network.py   #   MLP + Adam（relu/elu/tanh）
│   ├── format.py    #   自有模型格式
│   ├── ue_nmn.py    #   引擎 .nmn / 内嵌字节码模型读写与执行
│   └── onnx_io.py   #   最小 ONNX 读取器 + numpy 执行器
└── tests/           # 核心单测 + Blender 冒烟
```

## 许可

MIT，见 `LICENSE`。代码为本项目原创实现；文件中出现的格式常量（magic、版本号、层级编号等）属于互操作性所需的事实信息，来源于引擎运行时公开的文件布局。
