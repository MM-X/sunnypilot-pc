import onnx
from onnx import helper, numpy_helper, TensorProto, ValueInfoProto
from pathlib import Path
import numpy as np
import sys

# --- 配置 ---
INPUT_ONNX_PATH = Path(__file__).parent / '../models/driving_vision_fp16.onnx'
OUTPUT_ONNX_PATH = Path(__file__).parent / '../models/driving_vision_fp16_clipped.onnx'
CLIP_MIN = -65504.0
CLIP_MAX = 65504.0
# --- 配置结束 ---

def add_clip_after_mul(input_path: Path, output_path: Path, clip_min: float, clip_max: float):
    if not input_path.exists():
        print(f"错误：输入模型文件未找到: {input_path}")
        sys.exit(1)

    print(f"正在加载模型: {input_path}")
    try:
        model = onnx.load(str(input_path), load_external_data=False)
        graph = model.graph
        # 移除旧的形状信息，因为我们要修改图结构
        graph.ClearField("value_info")
        print("模型加载成功，已清除旧的 ValueInfo。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        sys.exit(1)

    # 构建现有输入/输出/初始化器的查找映射
    value_info_all = {vi.name: vi for vi in graph.value_info} # Should be empty now
    for inp in graph.input: value_info_all[inp.name] = inp
    for outp in graph.output: value_info_all[outp.name] = outp
    initializer_names = {init.name for init in graph.initializer}
    node_name_set = {node.name for node in graph.node if node.name} # Track existing node names

    new_nodes = []
    initializers_to_add = {}
    connection_updates = {} # original_mul_output -> new_clip_output
    value_info_to_add = {} # Store new ValueInfo protos for clip outputs

    # --- 创建 Clip 常量 ---
    clip_min_const_name = "global_clip_min_fp16"
    clip_max_const_name = "global_clip_max_fp16"
    if clip_min_const_name not in initializer_names:
        min_tensor = numpy_helper.from_array(np.array(clip_min, dtype=np.float16), name=clip_min_const_name)
        initializers_to_add[clip_min_const_name] = min_tensor
        initializer_names.add(clip_min_const_name) # Add to set
    if clip_max_const_name not in initializer_names:
        max_tensor = numpy_helper.from_array(np.array(clip_max, dtype=np.float16), name=clip_max_const_name)
        initializers_to_add[clip_max_const_name] = max_tensor
        initializer_names.add(clip_max_const_name) # Add to set

    print("\n--- 1. 构建新的节点列表并插入 Clip ---")
    mul_count = 0
    for node in graph.node:
        # 首先将原始节点添加到新列表
        new_nodes.append(node)

        if node.op_type == 'Mul':
            mul_count += 1
            if len(node.output) != 1:
                print(f"警告：Mul 节点 '{node.name}' 输出数量不为 1，跳过添加 Clip。")
                continue

            mul_output_name = node.output[0]

            # --- 生成唯一的 Clip 节点和输出名称 ---
            clip_node_name = node.name + "_clip" if node.name else "clip_" + mul_output_name
            original_clip_node_name = clip_node_name
            suffix = 1
            while clip_node_name in node_name_set:
                 clip_node_name = f"{original_clip_node_name}_{suffix}"
                 suffix += 1
            node_name_set.add(clip_node_name) # Add new name to set

            clip_output_name = mul_output_name + "_clipped"
            original_clip_output_name = clip_output_name
            suffix = 1
            # Check against all known tensor names (inputs, outputs, initializers, and newly created clip outputs)
            while clip_output_name in value_info_all or clip_output_name in initializer_names or clip_output_name in connection_updates.values():
                 clip_output_name = f"{original_clip_output_name}_{suffix}"
                 suffix += 1
            # --- 名称生成结束 ---

            print(f"  处理 Mul 节点: '{node.name}' (输出: '{mul_output_name}')")
            print(f"    插入 Clip 节点: '{clip_node_name}' (输出: '{clip_output_name}')")

            # 创建 Clip 节点
            clip_node = helper.make_node(
                'Clip',
                inputs=[mul_output_name, clip_min_const_name, clip_max_const_name], # 确保输入是 Mul 的输出
                outputs=[clip_output_name], # 确保输出是新名称
                name=clip_node_name
            )
            # 将 Clip 节点紧随 Mul 节点添加到新列表
            new_nodes.append(clip_node)

            # 记录连接更新
            connection_updates[mul_output_name] = clip_output_name

            # 尝试为 Clip 输出创建 ValueInfo (类型与 Mul 输出相同，形状未知)
            # 形状推断应该能填充正确的形状
            clip_vi = ValueInfoProto()
            clip_vi.name = clip_output_name
            # 尝试查找 Mul 输出的类型信息 (可能在 graph.output 或需要推断)
            # 暂时只设置名字，让形状推断处理类型和形状
            # if mul_output_name in value_info_all:
            #    if value_info_all[mul_output_name].type.tensor_type.elem_type:
            #        clip_vi.type.CopyFrom(value_info_all[mul_output_name].type)
            #        clip_vi.type.tensor_type.ClearField("shape") # Clear shape, let inference fill it
            value_info_to_add[clip_output_name] = clip_vi


    if mul_count == 0:
        print("未找到任何 Mul 节点，无需修改。")
        # sys.exit(0) # Or copy file

    print(f"处理了 {mul_count} 个 Mul 节点，插入了 {mul_count} 个 Clip 节点。")

    # --- 2. 替换旧节点列表，添加初始化器 ---
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(initializers_to_add.values())
    print(f"更新了图节点列表，添加了 {len(initializers_to_add)} 个常量。")

    # --- 3. 更新所有节点的输入连接 ---
    print("\n--- 3. 更新所有节点的输入连接 ---")
    updated_connections_count = 0
    for node in graph.node:
        # 特别注意：不要修改 Clip 节点自身的输入！
        # Clip 节点的输入已经在创建时设置好了 ([mul_output_name, min_const, max_const])
        # 我们只需要修改那些 *使用* 了原始 Mul 输出的节点
        if node.op_type == 'Clip' and node.name.endswith("_clip"): # Basic check if it's one of our added clips
             # print(f"  跳过更新 Clip 节点 '{node.name}' 的输入。")
             continue

        inputs_changed_in_node = False
        for i in range(len(node.input)):
            input_name = node.input[i]
            if input_name in connection_updates:
                new_input_name = connection_updates[input_name]
                # print(f"  节点 '{node.name}' (类型: {node.op_type}) 的输入 '{input_name}' 更新为 '{new_input_name}'")
                node.input[i] = new_input_name
                inputs_changed_in_node = True
        if inputs_changed_in_node:
            updated_connections_count += 1

    # --- 4. 更新图的输出 ---
    outputs_changed = False
    new_graph_outputs = []
    for output_vi in graph.output:
        if output_vi.name in connection_updates:
            new_output_name = connection_updates[output_vi.name]
            print(f"更新图输出 '{output_vi.name}' 为 Clip 输出 '{new_output_name}'")
            # 使用我们之前创建的 ValueInfoProto 占位符
            new_output_vi = value_info_to_add.get(new_output_name, ValueInfoProto()) # Get placeholder or create empty
            new_output_vi.name = new_output_name # Ensure name is correct
            # 尝试复制原始输出的类型信息（如果存在）
            if output_vi.type.tensor_type.elem_type:
                 new_output_vi.type.CopyFrom(output_vi.type)
                 new_output_vi.type.tensor_type.ClearField("shape") # Clear shape for inference

            new_graph_outputs.append(new_output_vi)
            outputs_changed = True
        else:
            new_graph_outputs.append(output_vi)

    del graph.output[:]
    graph.output.extend(new_graph_outputs)

    print(f"更新了 {updated_connections_count} 个节点的输入连接。")
    if outputs_changed:
        print("更新了图的输出定义。")

    # --- 5. 最终检查和形状推断 ---
    print("\n--- 5. 最终检查和形状推断 ---")
    try:
        # 运行形状推断来填充所有 ValueInfo，包括新 Clip 的输出
        print("运行最终形状推断...")
        # 确保模型 opset version 足够支持 Clip (opset 11+)
        if model.opset_import[0].version < 11:
             print(f"警告：模型 Opset 版本 ({model.opset_import[0].version}) 低于 11。将尝试升级到 11 以支持 Clip(min, max) 输入。")
             model.opset_import[0].version = 11

        # 运行形状推断
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True) # Use strict_mode=False initially
        print("最终形状推断完成。")

        # 再次检查模型
        onnx.checker.check_model(model)
        print("模型检查通过。")

    except Exception as e:
        print(f"警告：在检查或最终形状推断时发生错误: {e}")
        print("模型可能仍能保存，但可能无效。请仔细检查错误信息。")
        # sys.exit(1)

    # --- 6. 保存修改后的模型 ---
    print(f"\n--- 6. 正在保存修改后的模型到: {output_path} ---")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_path))
        print("模型保存成功。")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    add_clip_after_mul(INPUT_ONNX_PATH, OUTPUT_ONNX_PATH, CLIP_MIN, CLIP_MAX)
    print("\n脚本执行完毕。")
    print(f"修改后的模型已保存到: {OUTPUT_ONNX_PATH}")
    print("请再次尝试使用这个新模型文件进行 RKNN 转换。")
