import onnx
from onnx import helper, TensorProto
from pathlib import Path
import sys
import copy

# --- 配置 ---
# 输入 ONNX 模型路径
INPUT_ONNX_PATH = Path(__file__).parent / '../models/driving_vision.onnx'
# 输出 ONNX 模型路径 (修改后)
OUTPUT_ONNX_PATH = Path(__file__).parent / '../models/driving_vision_fp16.onnx'

# 需要修改类型的输入名称列表
INPUTS_TO_MODIFY = ['input_imgs', 'big_input_imgs']
# 目标数据类型
TARGET_DTYPE = TensorProto.DataType.FLOAT16
# --- 配置结束 ---

def get_type_name(dtype_enum):
  """将 ONNX DataType 枚举值转换为可读名称"""
  try:
    return onnx.TensorProto.DataType.Name(dtype_enum)
  except ValueError:
    return f"未知类型 ({dtype_enum})"

def modify_model(input_path: Path, output_path: Path, inputs_to_modify: list, target_dtype: int):
    """加载ONNX模型，修改输入类型，移除Cast，重连Concat"""
    if not input_path.exists():
        print(f"错误：输入模型文件未找到: {input_path}")
        sys.exit(1)

    print(f"正在加载模型: {input_path}")
    try:
        model = onnx.load(str(input_path))
        graph = model.graph
        print("模型加载成功.")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        sys.exit(1)

    nodes_to_remove = []
    cast_output_to_original_input = {} # 映射 Cast 输出名 -> 原始输入名

    print(f"\n--- 1. 修改输入类型为 {get_type_name(target_dtype)} ---")
    inputs_modified_count = 0
    for i in range(len(graph.input)):
        inp = graph.input[i]
        if inp.name in inputs_to_modify:
            original_dtype_name = get_type_name(inp.type.tensor_type.elem_type)
            print(f"找到输入 '{inp.name}'，原始类型: {original_dtype_name}。正在修改为 {get_type_name(target_dtype)}...")
            inp.type.tensor_type.elem_type = target_dtype
            inputs_modified_count += 1
    if inputs_modified_count < len(inputs_to_modify):
        print(f"警告：只找到了 {inputs_modified_count} 个需要修改的输入（共 {len(inputs_to_modify)} 个）。")
    elif inputs_modified_count == 0:
        print(f"警告：未找到任何需要修改的输入 {inputs_to_modify}。")
        # 脚本可能无法按预期工作，但继续尝试查找 Cast
        pass # 继续执行，也许用户只想移除Cast？

    print("\n--- 2. 查找紧随输入的 Cast 节点 ---")
    for node in graph.node:
        if node.op_type == 'Cast' and len(node.input) == 1 and node.input[0] in inputs_to_modify:
            original_input_name = node.input[0]
            cast_output_name = node.output[0]
            print(f"找到 Cast 节点 '{node.name}'，输入: '{original_input_name}'，输出: '{cast_output_name}'。标记待移除。")
            nodes_to_remove.append(node)
            cast_output_to_original_input[cast_output_name] = original_input_name

    if not nodes_to_remove:
        print("警告：未找到连接到指定输入的 Cast 节点。请检查模型结构。")
        # 如果没有 Cast，可能不需要重连 Concat，但还是检查一下
        # sys.exit(1) # 或者直接退出

    print("\n--- 3. 重定向后续节点 (预期是 Concat) 的输入 ---")
    concat_node_found = False
    for node in graph.node:
        # 检查当前节点的输入是否包含任何一个 Cast 节点的输出
        inputs_changed = False
        original_inputs = list(node.input) # 复制一份用于迭代时修改
        for i in range(len(original_inputs)):
            input_name = original_inputs[i]
            if input_name in cast_output_to_original_input:
                original_input_name = cast_output_to_original_input[input_name]
                print(f"找到节点 '{node.name}' (类型: {node.op_type}) 使用了 Cast 输出 '{input_name}'。")
                print(f"  将其输入重定向到原始输入 '{original_input_name}'。")
                # 直接修改 node.input 列表中的元素
                node.input[i] = original_input_name
                inputs_changed = True
                if node.op_type == 'Concat':
                    concat_node_found = True # 确认我们找到了预期的 Concat

        if inputs_changed and not concat_node_found:
             print(f"警告：节点 '{node.name}' (类型: {node.op_type}) 的输入被重定向了，但它不是 Concat。请确认这是否符合预期。")


    if not concat_node_found and nodes_to_remove: # 只有在确实移除了 Cast 时才需要 Concat
         print("警告：未找到预期中使用 Cast 输出的 Concat 节点。请手动检查模型图，确保后续连接正确。")


    print("\n--- 4. 移除 Cast 节点 ---")
    if nodes_to_remove:
        for node_to_remove in nodes_to_remove:
            print(f"正在移除 Cast 节点 '{node_to_remove.name}'")
            graph.node.remove(node_to_remove)
    else:
        print("没有需要移除的 Cast 节点。")

    # --- 5. (可选但推荐) 清理 ValueInfo ---
    print("\n--- 5. 清理 ValueInfo ---")
    value_info_to_remove_names = set(cast_output_to_original_input.keys())
    if value_info_to_remove_names:
        # 创建新的 ValueInfo 列表，排除掉被移除 Cast 的输出
        original_value_info = list(graph.value_info) # 获取当前列表的副本
        new_value_info = [vi for vi in original_value_info if vi.name not in value_info_to_remove_names]

        # --- 修改开始 ---
        # 删除旧的 ValueInfo 条目 (UPB 不支持 clear，所以我们重新构建)
        # 首先删除所有现有的条目
        while len(graph.value_info) > 0:
            graph.value_info.pop()
        # 然后添加过滤后的条目
        graph.value_info.extend(new_value_info)
        # --- 修改结束 ---

        print(f"已从 ValueInfo 中移除 {len(value_info_to_remove_names)} 个与 Cast 输出相关的条目，并重建了 ValueInfo 列表。")
    else:
        print("没有需要从 ValueInfo 中移除的条目。")


    # --- 6. 形状和类型推断 ---
    print("\n--- 6. 运行形状和类型推断 ---")
    try:
        # 在修改后重新进行形状和类型推断是个好习惯
        model = onnx.shape_inference.infer_shapes(model)
        print("形状推断完成。")
    except Exception as e:
        print(f"警告：形状推断失败，但这可能不影响模型的保存: {e}")

    # --- 7. 保存修改后的模型 ---
    print(f"\n--- 7. 正在保存修改后的模型到: {output_path} ---")
    try:
        onnx.save(model, str(output_path))
        print("模型保存成功。")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    modify_model(INPUT_ONNX_PATH, OUTPUT_ONNX_PATH, INPUTS_TO_MODIFY, TARGET_DTYPE)
    print("\n模型修改完成。请使用新的模型文件进行后续操作：")
    print(OUTPUT_ONNX_PATH)