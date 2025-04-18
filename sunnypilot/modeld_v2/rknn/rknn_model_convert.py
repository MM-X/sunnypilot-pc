import numpy as np
from pathlib import Path
from rknn.api import RKNN


ONNX_PATH = Path(__file__).parent / '../models/supercombo.onnx'
RKNN_PATH = Path(__file__).parent / '../models/supercombo.rknn'

ONNX_MODEL = [ONNX_PATH]
RKNN_MODEL = [RKNN_PATH]


if __name__ == '__main__':
    for onnx_model, rknn_model in zip(ONNX_MODEL, RKNN_MODEL):
      rknn = RKNN(verbose=False, verbose_file='./rknn_model_convert.log')

      # pre-process config
      print('--> Config model')
      rknn.config(target_platform='rk3588', optimization_level=0)
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
      ret = rknn.build( do_quantization=False )
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

