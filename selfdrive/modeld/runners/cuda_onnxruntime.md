确保安装的nvidia驱动是cuda12
sudo apt update
sudo apt install nvidia-cuda-toolkit
参考https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cudnn/9.7.1/local_installers/cudnn-local-repo-ubuntu2404-9.7.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.7.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.7.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12


cd sunnypilot-pc/
source .venv/bin/active
cd .venv/lib
cd .venv/lib/python3.12/site-packages/
ls | grep onnxruntime
rm rf onnxruntim*
cd sunnypilot-pc/.venv
pip3 install --upgrade  onnxruntime-gpu --prefix=./ --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

然后看下安装成功没
cd sunnypilot-pc/
source .venv/bin/active
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
输出下面就代表安装成功
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']


然后改代码ort_helpers.py，最后一行删掉，改称
provs = ['CUDAExecutionProvider', 'CPUExecutionProvider']
return ort.InferenceSession(model_data,  options, providers=provs)

再运行应该就能看到推理速度很快
Enjoy！！！

