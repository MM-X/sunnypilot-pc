import onnx
from onnx import helper, numpy_helper, TensorProto, ValueInfoProto
from pathlib import Path
import numpy as np
import sys

# --- 配置 ---
# 输入 ONNX 模型路径 (通常是 Mul 后面加了 Clip 的版本)
INPUT_ONNX_PATH = Path(__file__).parent / '../models/driving_vision_fp16_clipped.onnx'
# 输出 ONNX 模型路径 (进一步稳定化后)
OUTPUT_ONNX_PATH = Path(__file__).parent / '../models/driving_vision_fp16_stabilized.onnx' #保持这个输出文件名

# 目标节点/张量名称 (根据你的 error_analysis.txt 调整)
# *** 关键：找到产生 inf 的节点/输出 ***
REDUCESUM_INF_OUTPUT_NAME = "/summarizer/ReduceSum_output_0" # 产生 inf 的输出张量名
SQRT_NODE_NAME = "/summarizer/Sqrt_output_0" # 后续 Sqrt 节点的名字或其输出张量的名字
DIV_NODE_NAME = "/summarizer/Div_output_0"   # 后续 Div 节点的名字或其输出张量的名字

# Clip ReduceSum 输出的范围 (FP16)
REDUCESUM_CLIP_MIN = -65504.0
REDUCESUM_CLIP_MAX = 65504.0
# Clip Sqrt 输入的最小值
SQRT_CLIP_MIN = 0.0
# 加到 Div 除数上的 Epsilon (FP16)
DIV_EPSILON_FP16 = np.float16(1e-7)
# --- 配置结束 ---

def find_node_by_output(graph, output_name):
    for node in graph.node:
        if output_name in node.output:
            return node
    return None

def find_node_by_name(graph, node_name):
     for node in graph.node:
         if node.name == node_name:
             return node
     return None

# Helper to generate unique names
def generate_unique_name(base_name, existing_names_set):
    name = base_name
    suffix = 1
    while name in existing_names_set:
        name = f"{base_name}_{suffix}"
        suffix += 1
    existing_names_set.add(name)
    return name

def stabilize_summarizer(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"错误：输入模型文件未找到: {input_path}")
        sys.exit(1)

    print(f"正在加载模型: {input_path}")
    try:
        model = onnx.load(str(input_path), load_external_data=False)
        graph = model.graph
        graph.ClearField("value_info")
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        sys.exit(1)

    # --- 查找目标节点 ---
    reduce_sum_node = find_node_by_output(graph, REDUCESUM_INF_OUTPUT_NAME)
    if not reduce_sum_node:
        print(f"错误：找不到产生输出 '{REDUCESUM_INF_OUTPUT_NAME}' 的节点。请检查名称。")
        # 尝试通过名字查找（如果名称和输出名一样）
        reduce_sum_node = find_node_by_name(graph, REDUCESUM_INF_OUTPUT_NAME)
        if not reduce_sum_node:
             print(f"错误：也找不到名为 '{REDUCESUM_INF_OUTPUT_NAME}' 的节点。")
             sys.exit(1)
        # 确认找到的节点确实输出了这个张量
        if REDUCESUM_INF_OUTPUT_NAME not in reduce_sum_node.output:
             print(f"错误：节点 '{reduce_sum_node.name}' 不输出 '{REDUCESUM_INF_OUTPUT_NAME}'。")
             sys.exit(1)


    sqrt_node = find_node_by_output(graph, SQRT_NODE_NAME) or find_node_by_name(graph, SQRT_NODE_NAME)
    div_node = find_node_by_output(graph, DIV_NODE_NAME) or find_node_by_name(graph, DIV_NODE_NAME)

    if not sqrt_node: print(f"警告：找不到 Sqrt 节点 '{SQRT_NODE_NAME}'。"); sqrt_node = None # Allow continuing
    if not div_node: print(f"警告：找不到 Div 节点 '{DIV_NODE_NAME}'。"); div_node = None # Allow continuing
    if div_node and div_node.op_type != 'Div': print(f"警告：节点 '{div_node.name}' 不是 Div 类型。"); div_node = None
    if sqrt_node and sqrt_node.op_type != 'Sqrt': print(f"警告：节点 '{sqrt_node.name}' 不是 Sqrt 类型。"); sqrt_node = None

    print(f"找到产生 inf 的节点: '{reduce_sum_node.name}' (类型: {reduce_sum_node.op_type}), 输出: '{REDUCESUM_INF_OUTPUT_NAME}'")
    if sqrt_node: print(f"找到 Sqrt 节点: '{sqrt_node.name}'")
    if div_node: print(f"找到 Div 节点: '{div_node.name}'")

    new_nodes = []
    initializers_to_add = {}
    connection_updates = {} # original_output -> new_output
    node_name_set = {node.name for node in graph.node if node.name}
    tensor_name_set = {init.name for init in graph.initializer} | {inp.name for inp in graph.input} | {outp.name for outp in graph.output}

    # --- 创建常量 ---
    # Epsilon for Div
    epsilon_const_name = generate_unique_name("global_div_epsilon_fp16", tensor_name_set)
    epsilon_tensor = numpy_helper.from_array(np.array(DIV_EPSILON_FP16), name=epsilon_const_name)
    initializers_to_add[epsilon_const_name] = epsilon_tensor
    # Min=0 for Sqrt input clip
    sqrt_clip_min_const_name = generate_unique_name("global_sqrt_clip_min_fp16", tensor_name_set)
    min_0_tensor = numpy_helper.from_array(np.array(SQRT_CLIP_MIN, dtype=np.float16), name=sqrt_clip_min_const_name)
    initializers_to_add[sqrt_clip_min_const_name] = min_0_tensor
    # Min/Max for ReduceSum output clip
    rs_clip_min_const_name = generate_unique_name("global_rs_clip_min_fp16", tensor_name_set)
    rs_clip_max_const_name = generate_unique_name("global_rs_clip_max_fp16", tensor_name_set)
    rs_min_tensor = numpy_helper.from_array(np.array(REDUCESUM_CLIP_MIN, dtype=np.float16), name=rs_clip_min_const_name)
    rs_max_tensor = numpy_helper.from_array(np.array(REDUCESUM_CLIP_MAX, dtype=np.float16), name=rs_clip_max_const_name)
    initializers_to_add[rs_clip_min_const_name] = rs_min_tensor
    initializers_to_add[rs_clip_max_const_name] = rs_max_tensor


    # --- 1. 构建新节点列表，插入修改 ---
    modified_reducesum_output = False
    modified_sqrt_input = False
    modified_div_divisor = False

    for node in graph.node:
        current_node_outputs = list(node.output) # Outputs of the node *before* potential modifications below

        # --- 1. Clip ReduceSum output ---
        if node.name == reduce_sum_node.name:
            print(f"处理 ReduceSum/Conv 节点 '{node.name}'...")
            new_nodes.append(node) # Add the original node first

            # Create Clip node for its output
            clip_rs_node_name = generate_unique_name(node.name + "_output_clip", node_name_set)
            clip_rs_output_name = generate_unique_name(REDUCESUM_INF_OUTPUT_NAME + "_clipped_fp16range", tensor_name_set)

            print(f"  插入 Clip (FP16范围) 节点 '{clip_rs_node_name}' after '{node.name}'")
            clip_rs_node = helper.make_node(
                'Clip',
                inputs=[REDUCESUM_INF_OUTPUT_NAME, rs_clip_min_const_name, rs_clip_max_const_name], # input, min, max
                outputs=[clip_rs_output_name],
                name=clip_rs_node_name
            )
            new_nodes.append(clip_rs_node) # Add the new Clip node
            connection_updates[REDUCESUM_INF_OUTPUT_NAME] = clip_rs_output_name # Record the output change
            modified_reducesum_output = True
            continue # Skip adding the original node again

        # --- 2. Clip Sqrt input ---
        elif sqrt_node and node.name == sqrt_node.name:
            print(f"处理 Sqrt 节点 '{node.name}'...")
            sqrt_original_input = node.input[0]
            # Check if Sqrt's input needs update from previous step
            sqrt_input_to_clip = connection_updates.get(sqrt_original_input, sqrt_original_input)

            # Create Clip node (min=0) for Sqrt's input
            clip_sqrt_input_node_name = generate_unique_name(node.name + "_input_clip", node_name_set)
            clip_sqrt_input_output_name = generate_unique_name(sqrt_original_input + "_clipped_min0", tensor_name_set)

            print(f"  插入 Clip (min=0) 节点 '{clip_sqrt_input_node_name}' before Sqrt")
            clip_sqrt_input_node = helper.make_node(
                'Clip',
                inputs=[sqrt_input_to_clip, sqrt_clip_min_const_name], # input, min
                outputs=[clip_sqrt_input_output_name],
                name=clip_sqrt_input_node_name
            )
            new_nodes.append(clip_sqrt_input_node) # Add Clip node before Sqrt

            # Modify Sqrt node's input
            node.input[0] = clip_sqrt_input_output_name
            new_nodes.append(node) # Add the modified Sqrt node
            modified_sqrt_input = True
            continue

        # --- 3. Add Epsilon to Div divisor ---
        elif div_node and node.name == div_node.name:
            print(f"处理 Div 节点 '{node.name}'...")
            div_original_divisor = node.input[1]
            # Check if Div's divisor needs update from previous steps
            divisor_to_modify = connection_updates.get(div_original_divisor, div_original_divisor)

            # Create Add node for the divisor
            add_epsilon_node_name = generate_unique_name(node.name + "_add_epsilon", node_name_set)
            add_epsilon_output_name = generate_unique_name(div_original_divisor + "_plus_epsilon", tensor_name_set)

            print(f"  插入 Add Epsilon 节点 '{add_epsilon_node_name}' for Divisor")
            add_epsilon_node = helper.make_node(
                'Add',
                inputs=[divisor_to_modify, epsilon_const_name],
                outputs=[add_epsilon_output_name],
                name=add_epsilon_node_name
            )
            new_nodes.append(add_epsilon_node) # Add Add node before Div

            # Modify Div node's divisor input
            node.input[1] = add_epsilon_output_name
            # Also update the dividend if needed
            div_original_dividend = node.input[0]
            node.input[0] = connection_updates.get(div_original_dividend, div_original_dividend)

            new_nodes.append(node) # Add the modified Div node
            modified_div_divisor = True
            continue

        # --- Default: Add node, potentially updating its inputs ---
        else:
            inputs_changed = False
            for i in range(len(node.input)):
                if node.input[i] in connection_updates:
                    node.input[i] = connection_updates[node.input[i]]
                    inputs_changed = True
            # if inputs_changed: print(f"  更新了节点 '{node.name}' 的输入。")
            new_nodes.append(node)


    if not modified_reducesum_output: print(f"警告：未修改 ReduceSum/Conv 节点 '{reduce_sum_node.name}' 的输出。")
    if sqrt_node and not modified_sqrt_input: print(f"警告：未修改 Sqrt 节点 '{sqrt_node.name}' 的输入。")
    if div_node and not modified_div_divisor: print(f"警告：未修改 Div 节点 '{div_node.name}' 的除数输入。")

    # --- 2. 替换节点列表和添加初始化器 ---
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(initializers_to_add.values())
    print(f"更新了图节点列表，添加了 {len(initializers_to_add)} 个常量。")

    # --- 3. 更新图输出 ---
    outputs_changed = False
    new_graph_outputs = []
    for output_vi in graph.output:
        original_output_name = output_vi.name
        if original_output_name in connection_updates:
             new_output_name = connection_updates[original_output_name]
             print(f"更新图输出 '{original_output_name}' 为 '{new_output_name}'")
             # Create new ValueInfoProto, try to copy type
             new_output_vi = ValueInfoProto()
             new_output_vi.name = new_output_name
             if output_vi.type.tensor_type.elem_type:
                  new_output_vi.type.CopyFrom(output_vi.type)
                  new_output_vi.type.tensor_type.ClearField("shape")
             new_graph_outputs.append(new_output_vi)
             outputs_changed = True
        else:
             new_graph_outputs.append(output_vi)
    if outputs_changed:
        del graph.output[:]
        graph.output.extend(new_graph_outputs)
        print("更新了图的输出定义。")


    # --- 4. 最终检查和形状推断 ---
    print("\n--- 4. 最终检查和形状推断 ---")
    try:
        print("运行最终形状推断...")
        if model.opset_import[0].version < 11:
             print(f"警告：模型 Opset 版本 ({model.opset_import[0].version}) 低于 11。升级到 11。")
             model.opset_import[0].version = 11
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        print("最终形状推断完成。")
        onnx.checker.check_model(model)
        print("模型检查通过。")
    except Exception as e:
        print(f"警告：在检查或最终形状推断时发生错误: {e}")

    # --- 5. 保存模型 ---
    print(f"\n--- 5. 正在保存修改后的模型到: {output_path} ---")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_path))
        print("模型保存成功。")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    stabilize_summarizer(INPUT_ONNX_PATH, OUTPUT_ONNX_PATH)
    print("\n脚本执行完毕。")
    print(f"修改后的模型已保存到: {OUTPUT_ONNX_PATH}")
    print("请再次使用这个新模型文件 'driving_vision_fp16_stabilized.onnx' 进行 RKNN 转换。")
