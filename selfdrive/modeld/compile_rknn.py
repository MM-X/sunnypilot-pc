#!/usr/bin/env python3
"""Compile driving_supercombo.onnx -> driving_supercombo.rknn for RK3588 NPU.

FP16, no quantization. Inputs are preprocessed tensors (YUV stack, not RGB),
so no mean/std normalization. Output is a single fused tensor sliced at runtime.

Usage:
  python3 selfdrive/modeld/compile_rknn.py
"""
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
ONNX_PATH = MODELS_DIR / "driving_supercombo.onnx"
RKNN_PATH = MODELS_DIR / "driving_supercombo.rknn"
TARGET_PLATFORM = "rk3588"


def _print_io(onnx_path: Path):
  import onnx
  m = onnx.load(str(onnx_path))
  for label, graph in (("INPUTS", m.graph.input), ("OUTPUTS", m.graph.output)):
    print(f"  {label}:")
    for t in graph:
      dims = [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]
      print(f"    {t.name}: {dims}")


def _normalize_for_rknn(graph):
  """Expand Where(bool-const cond) into arithmetic and cast GatherND int64
  indices to int32 — RKNN's subgraph fallback aborts on these dtypes."""
  import numpy as np
  import onnx
  from onnx import helper, numpy_helper
  inits = {i.name: i for i in graph.initializer}
  new_nodes, n_where, n_gathernd = [], 0, 0
  for node in graph.node:
    if node.op_type == "Where" and len(node.input) == 3 and node.input[0] in inits:
      cond_name, a, b = node.input
      cond_init = inits[cond_name]
      if cond_init.data_type != onnx.TensorProto.BOOL:
        new_nodes.append(node); continue
      out = node.output[0]
      cf = f"{out}_condf"
      cond_arr = numpy_helper.to_array(cond_init).astype(np.float16)
      new_nodes.append(helper.make_node("Constant", [], [cf], name=f"{out}_condf_c",
                                        value=numpy_helper.from_array(cond_arr)))
      one = f"{out}_one"
      new_nodes.append(helper.make_node("Constant", [], [one], name=f"{out}_one_c",
                                        value=numpy_helper.from_array(np.array(1.0, np.float16))))
      t1, t2, t3 = f"{out}_t1", f"{out}_t2", f"{out}_t3"
      # out = a*cond + b*(1-cond)
      new_nodes += [
        helper.make_node("Mul", [a, cf], [t1], name=f"{out}_mul1"),
        helper.make_node("Sub", [one, cf], [t2], name=f"{out}_sub"),
        helper.make_node("Mul", [b, t2], [t3], name=f"{out}_mul2"),
        helper.make_node("Add", [t1, t3], [out], name=f"{out}_add"),
      ]
      n_where += 1
    elif node.op_type == "GatherND" and len(node.input) >= 2 and node.input[1] in inits \
         and inits[node.input[1]].data_type == onnx.TensorProto.INT64:
      data, idx = node.input[0], node.input[1]
      idx32 = f"{idx}_i32"
      new_nodes.append(helper.make_node("Cast", [idx], [idx32], name=f"{idx}_cast_i32",
                                       to=onnx.TensorProto.INT32))
      new_inputs = [data, idx32] + list(node.input[2:])
      new_nodes.append(helper.make_node("GatherND", new_inputs, list(node.output),
                                        name=node.name))
      n_gathernd += 1
    else:
      new_nodes.append(node)
  del graph.node[:]; graph.node.extend(new_nodes)
  return n_where, n_gathernd


def _downgrade_opset(onnx_path: Path, max_opset: int = 19) -> Path:
  """RKNN toolkit2 needs opset <= 19. supercombo is opset 20 only because of
  Gelu (opset 20). Expand Gelu(tanh) into an erf-free subgraph of opset-19 ops
  (Mul/Add/Tanh/Constant), then relabel the opset. Other opset-20-only ops
  are absent from supercombo, so a plain relabel is safe."""
  import numpy as np
  import onnx
  from onnx import helper, numpy_helper

  m = onnx.load(str(onnx_path))
  cur = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
  if cur <= max_opset:
    return onnx_path
  print(f"  opset {cur} > {max_opset}, expanding Gelu(tanh) + relabeling...")

  C1 = 0.044715
  C2 = 0.7978845608028654  # sqrt(2/pi)
  new_nodes, gelu_n = [], 0
  for node in m.graph.node:
    if node.op_type != "Gelu":
      new_nodes.append(node); continue
    appr = next((a.s.decode() for a in node.attribute if a.name == "approximate"), "none")
    if appr != "tanh":
      raise ValueError(f"unsupported Gelu approximate={appr!r}; only 'tanh' handled")
    gelu_n += 1
    ns = f"gelu{gelu_n}"
    x, y = node.input[0], node.output[0]
    def const(name, val):
      return helper.make_node("Constant", [], [name], name=f"{name}_c",
                              value=numpy_helper.from_array(np.array(val, np.float32)))
    x2, x3 = f"{ns}_x2", f"{ns}_x3"
    t1, inner = f"{ns}_t1", f"{ns}_inner"
    t2, t = f"{ns}_t2", f"{ns}_t"
    half_x, onept = f"{ns}_halfx", f"{ns}_1pt"
    c1, c2, c3, one = f"{ns}_c1", f"{ns}_c2", f"{ns}_c3", f"{ns}_one"
    # inner = C1*x^3 + x ; t = tanh(C2*inner) ; y = 0.5*x*(1+t)
    new_nodes += [
      const(c1, C1), const(c2, C2), const(c3, 0.5), const(one, 1.0),
      helper.make_node("Mul", [x, x], [x2], name=f"{ns}_x2"),
      helper.make_node("Mul", [x2, x], [x3], name=f"{ns}_x3"),
      helper.make_node("Mul", [c1, x3], [t1], name=f"{ns}_c1x3"),
      helper.make_node("Add", [t1, x], [inner], name=f"{ns}_inner"),
      helper.make_node("Mul", [c2, inner], [t2], name=f"{ns}_t2"),
      helper.make_node("Tanh", [t2], [t], name=f"{ns}_tanh"),
      helper.make_node("Mul", [c3, x], [half_x], name=f"{ns}_halfx"),
      helper.make_node("Add", [one, t], [onept], name=f"{ns}_1pt"),
      helper.make_node("Mul", [half_x, onept], [y], name=f"{ns}_out"),
    ]
  del m.graph.node[:]; m.graph.node.extend(new_nodes)
  nw, ng = _normalize_for_rknn(m.graph)
  if nw or ng:
    print(f"  normalized {nw} Where(bool) -> arith, {ng} GatherND int64 -> int32")
  for o in m.opset_import:
    if o.domain in ("", "ai.onnx"):
      o.version = max_opset
  out = onnx_path.with_suffix(".opset19.onnx")
  onnx.save(m, str(out))
  print(f"  expanded {gelu_n} Gelu(tanh), wrote {out}")
  return out


def main():
  if not ONNX_PATH.exists():
    print(f"❌ ONNX not found: {ONNX_PATH}", file=sys.stderr)
    return 1
  print(f"ONNX: {ONNX_PATH} ({ONNX_PATH.stat().st_size // 1024} KiB)")
  _print_io(ONNX_PATH)
  onnx_path = _downgrade_opset(ONNX_PATH)

  from rknn.api import RKNN
  rknn = RKNN(verbose=True)

  # float_dtype=fp16, no quantization (quantized_* params stay default-off).
  ret = rknn.config(target_platform=TARGET_PLATFORM, optimization_level=2, float_dtype='float16')
  if ret:
    print(f"❌ config failed: {ret}", file=sys.stderr); return 1

  ret = rknn.load_onnx(model=str(onnx_path))
  if ret:
    print(f"❌ load_onnx failed: {ret}", file=sys.stderr); return 1

  ret = rknn.build(do_quantization=False)
  if ret:
    print(f"❌ build failed: {ret}", file=sys.stderr); return 1

  RKNN_PATH.parent.mkdir(parents=True, exist_ok=True)
  ret = rknn.export_rknn(str(RKNN_PATH))
  if ret:
    print(f"❌ export_rknn failed: {ret}", file=sys.stderr); return 1

  rknn.release()
  print(f"✅ {RKNN_PATH} ({RKNN_PATH.stat().st_size // 1024} KiB)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
