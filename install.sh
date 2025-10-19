
apt-get update
apt-get install -y libgl1 libglib2.0-0 libx11-6

pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmcv-full
pip install mamba-ssm==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm lmdb mmengine numpy==1.26.4 transformers==4.47.1 -i https://pypi.tuna.tsinghua.edu.cn/simple