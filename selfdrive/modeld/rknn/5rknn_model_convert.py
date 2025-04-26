import numpy as np
from pathlib import Path
from rknn.api import RKNN


VISION_ONNX_PATH = Path(__file__).parent / '../models/driving_vision_fp16.onnx'
POLICY_ONNX_PATH = Path(__file__).parent / '../models/driving_policy.onnx'
VISION_RKNN_PATH = Path(__file__).parent / '../models/driving_vision.rknn'
POLICY_RKNN_PATH = Path(__file__).parent / '../models/driving_policy.rknn'

ONNX_MODEL = [VISION_ONNX_PATH, POLICY_ONNX_PATH]
RKNN_MODEL = [VISION_RKNN_PATH, POLICY_RKNN_PATH]
DATA_SET = ['./vision_dataset.txt', './policy_dataset.txt']


if __name__ == '__main__':
    for onnx_model, rknn_model, dataset in zip(ONNX_MODEL, RKNN_MODEL, DATA_SET):
      rknn = RKNN(verbose=False, verbose_file='./rknn_model_convert.log')

      # pre-process config
      print('--> Config model')
      rknn.config(quantized_algorithm='kl_divergence', target_platform='rk3588', optimization_level=3, custom_string='v0.0.1')
      print('Config model done')

      # Load ONNX model
      print('--> Loading model')
      ret = rknn.load_onnx( model=str(onnx_model) )   #, inputs=['inputs', 'buffer'], input_size_list=[[514],[27724] ], outputs=['out', 'out_buffer'])

      if ret != 0:
          print('Load model failed!')
          exit(ret)
      print('Loading model done')

      # Build model
      print('--> Building model')
      ret = rknn.build( do_quantization=True, dataset=dataset )
      if ret != 0:
          print('Build model failed!')
          exit(ret)
      print('Building model done')

      # Export RKNN model
      print('--> Export rknn model')
      ret = rknn.export_rknn( str(rknn_model) )
      if ret != 0:
          print('Export rknn model failed!')
          exit(ret)
      print('Export rknn model done')

      # Release RKNN object
      rknn.release()
      print('Release RKNN done')
    print('All done!')

