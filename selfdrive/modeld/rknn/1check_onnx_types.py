import onnx
from pathlib import Path
import sys

# --- 配置 ---
# 将此路径替换为你的 ONNX 模型文件路径
ONNX_MODEL_PATH = Path(__file__).parent / '../models/driving_vision.onnx'
# ONNX_MODEL_PATH = Path('/home/mx/sunnypilot-pc/sunnypilot/modeld_v2/models/supercombo.onnx') # 或者直接指定绝对路径

# 定义可能不受 RKNN 支持的 ONNX 数据类型枚举值
# 参考 onnx.TensorProto.DataType:
# UNDEFINED = 0
# FLOAT = 1
# UINT8 = 2  <-- 错误日志中提到的 Dtype 2
# INT8 = 3
# UINT16 = 4
# INT16 = 5
# INT32 = 6
# INT64 = 7
# STRING = 8
# BOOL = 9
# FLOAT16 = 10
# DOUBLE = 11
# UINT32 = 12
# UINT64 = 13
# COMPLEX64 = 14
# COMPLEX128 = 15
# BFLOAT16 = 16
# FLOAT8... = 17+
# 通常 RKNN 对 FLOAT32 支持最好，FLOAT16 可能支持，INT8 需要量化流程
# 其他整数类型、BOOL、STRING、DOUBLE、COMPLEX 通常不支持
POTENTIALLY_UNSUPPORTED_DTYPES = {
    onnx.TensorProto.DataType.UINT8,    # 重点关注 Dtype 2
    onnx.TensorProto.DataType.INT8,     # INT8 通常需要量化
    onnx.TensorProto.DataType.UINT16,
    onnx.TensorProto.DataType.INT16,
    onnx.TensorProto.DataType.INT64,    # 64位整数可能不支持
    onnx.TensorProto.DataType.UINT32,
    onnx.TensorProto.DataType.UINT64,
    onnx.TensorProto.DataType.BOOL,
    onnx.TensorProto.DataType.STRING,
    onnx.TensorProto.DataType.DOUBLE,   # 64位浮点数通常不支持
    onnx.TensorProto.DataType.COMPLEX64,
    onnx.TensorProto.DataType.COMPLEX128,
    # 可以根据需要添加 FLOAT16 或 BFLOAT16，如果你的平台不支持它们
    # onnx.TensorProto.DataType.FLOAT16,
    # onnx.TensorProto.DataType.BFLOAT16,
}
# --- 配置结束 ---

def get_type_name(dtype_enum):
  """将 ONNX DataType 枚举值转换为可读名称"""
  try:
    return onnx.TensorProto.DataType.Name(dtype_enum)
  except ValueError:
    return f"未知类型 ({dtype_enum})"

def check_model_types(model_path: Path):
  """加载 ONNX 模型并检查不支持的数据类型"""
  if not model_path.exists():
    print(f"错误：模型文件未找到: {model_path}")
    sys.exit(1)

  print(f"正在加载模型: {model_path}")
  try:
    model = onnx.load(str(model_path))
    print("模型加载成功.")
    onnx.checker.check_model(model) # 基础检查
    print("ONNX 模型结构检查通过.")
  except Exception as e:
    print(f"加载或检查模型时出错: {e}")
    sys.exit(1)

  found_unsupported = False
  unsupported_details = []

  print("\n--- 检查模型输入 ---")
  for inp in model.graph.input:
    dtype = inp.type.tensor_type.elem_type
    type_name = get_type_name(dtype)
    print(f"输入: {inp.name}, 类型: {type_name} ({dtype})")
    if dtype in POTENTIALLY_UNSUPPORTED_DTYPES:
      found_unsupported = True
      unsupported_details.append(f"输入 '{inp.name}' 类型为 {type_name} ({dtype})")

  print("\n--- 检查模型输出 ---")
  for outp in model.graph.output:
    dtype = outp.type.tensor_type.elem_type
    type_name = get_type_name(dtype)
    print(f"输出: {outp.name}, 类型: {type_name} ({dtype})")
    if dtype in POTENTIALLY_UNSUPPORTED_DTYPES:
      found_unsupported = True
      unsupported_details.append(f"输出 '{outp.name}' 类型为 {type_name} ({dtype})")

  print("\n--- 检查初始化器 (权重/偏置等) ---")
  for initializer in model.graph.initializer:
    dtype = initializer.data_type
    type_name = get_type_name(dtype)
    # 通常权重很多，只打印可能不支持的
    if dtype in POTENTIALLY_UNSUPPORTED_DTYPES:
        print(f"初始化器: {initializer.name}, 类型: {type_name} ({dtype})")
        found_unsupported = True
        unsupported_details.append(f"初始化器 '{initializer.name}' 类型为 {type_name} ({dtype})")
    # else:
    #     print(f"Initializer: {initializer.name}, Type: {type_name} ({dtype})") # 取消注释以查看所有

  print("\n--- 检查中间值信息 (ValueInfo) ---")
  for value_info in model.graph.value_info:
      if value_info.type.HasField("tensor_type"):
          dtype = value_info.type.tensor_type.elem_type
          type_name = get_type_name(dtype)
          # 中间值也很多，只打印可能不支持的
          if dtype in POTENTIALLY_UNSUPPORTED_DTYPES:
              print(f"中间值: {value_info.name}, 类型: {type_name} ({dtype})")
              found_unsupported = True
              unsupported_details.append(f"中间值 '{value_info.name}' 类型为 {type_name} ({dtype})")
      # else:
      #     print(f"ValueInfo: {value_info.name}, Type: Non-tensor") # 其他类型

  print("\n--- 检查结果 ---")
  if found_unsupported:
    print("错误：在模型中检测到以下可能不受 RKNN 支持的数据类型:")
    for detail in unsupported_details:
      print(f"- {detail}")
    print("\n请使用 Netron 等工具详细检查模型结构，并确认 RKNN 对这些类型的支持情况。")
    print("可能需要修改模型来源或使用 RKNN 的量化功能（如果目标是 INT8）。")
  else:
    print("在检查的范围内未发现明显不受支持的数据类型。")
    print("注意：此脚本主要检查输入/输出/初始化器/ValueInfo 的类型，")
    print("某些算子本身可能对输入类型有限制，即使张量类型本身受支持。")
    print("如果构建仍然失败，请仔细检查 RKNN 构建日志以获取更详细的算子级错误信息。")

if __name__ == "__main__":
  check_model_types(ONNX_MODEL_PATH)