import numpy as np
from pathlib import Path
from rknn.api import RKNN

VISION_ONNX_PATH = Path(__file__).parent / './models/driving_vision_fp16_stabilized.onnx'
POLICY_ONNX_PATH = Path(__file__).parent / './models/driving_policy.onnx'
VISION_RKNN_PATH = Path(__file__).parent / './models/driving_vision.rknn'
POLICY_RKNN_PATH = Path(__file__).parent / './models/driving_policy.rknn'

rknn = RKNN(verbose=False, verbose_file='./rknn_model_convert.log')

rknn.config(quantized_algorithm='kl_divergence', target_platform='rk3588', optimization_level=0, custom_string='v0.0.1')

ret = rknn.load_onnx( model=str(VISION_ONNX_PATH) )

ret = rknn.build(do_quantization=False, dataset='./vision_dataset.txt')

input_imgs = np.load("./dataset/input_imgs_6.npy")
big_input_imgs = np.load("./dataset/big_input_imgs_6.npy")
desire = np.load("./dataset/desire_6.npy")
traffic_convention = np.load("./dataset/traffic_convention_6.npy")
lateral_control_params = np.load("./dataset/lateral_control_params_6.npy")
prev_desired_curv = np.load("./dataset/prev_desired_curv_6.npy")
features_buffer = np.load("./dataset/features_buffer_6.npy")
vision_inputs = [input_imgs, big_input_imgs]
policy_inputs = [desire, traffic_convention, lateral_control_params, prev_desired_curv, features_buffer]

ret = rknn.accuracy_analysis(inputs=vision_inputs, output_dir='./policy_snapshot_int8', target=None)

rknn.release()

