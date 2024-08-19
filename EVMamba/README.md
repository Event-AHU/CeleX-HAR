```shell
conda create -n evmamba python=3.10.13
conda activate evmamba
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

#### Please refer to Vmamba to install the following packages(https://github.com/MzeroMiko/VMamba)：
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
####

pip install -U openmim
mim install mmengine
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -v -e .
```