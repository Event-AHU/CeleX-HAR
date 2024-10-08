<div align="center">

<img src="https://github.com/Event-AHU/CeleX-HAR/blob/main/figures/celexhar_logo.png" width="500">
  
**Event Stream based Human Action Recognition: A High-Definition Benchmark Dataset and Algorithms**

------

</div>

# :dart: Abstract 
Human Action Recognition (HAR) stands as a pivotal research domain in both computer vision and artificial intelligence, with RGB cameras dominating as the preferred tool for investigation and innovation in this field. However, in real-world applications, RGB cameras encounter numerous challenges, including light conditions, fast motion, and privacy concerns. Consequently, bio-inspired event cameras have garnered increasing attention due to their advantages of low energy consumption, high dynamic range, etc. Nevertheless, most existing event-based HAR datasets are low resolution ($346 \times 260$). In this paper, we propose a large-scale, high-definition ($1280 \times 800$) human action recognition dataset based on the CeleX-V event camera, termed CeleX-HAR. It encompasses 150 commonly occurring action categories, comprising a total of 124,625 video sequences. Various factors such as multi-view, illumination, action speed, and occlusion are considered when recording these data. To build a more comprehensive benchmark dataset, we report over 20 mainstream HAR models for future works to compare. In addition, we also propose a novel Mamba vision backbone network for event stream based HAR, termed EVMamba, which equips the spatial plane multi-directional scanning and novel voxel temporal scanning mechanism. By encoding and mining the spatio-temporal information of event streams, our EVMamba has achieved favorable results across multiple datasets. Both the dataset and source code will be released upon acceptance. 


# :collision: Update Log


# :dvd: Demo Video 
A demo video can be found by clicking the image below: 
<p align="center">
  <a href="https://youtu.be/BaEbwVVuarw">
    <img src="https://github.com/Event-AHU/CeleX-HAR/blob/main/figures/CeleXHAR_youtube.png" alt="DemoVideo" width="800"/>
  </a>
</p>



<p align="center">
<img src="https://github.com/Event-AHU/CeleX-HAR/blob/main/CeleXHAR_samples.jpg" width="800">
</p>


# :hammer: Environment 

**A Spatial-Temporal Scanning framework for Event Stream-based Human Action Recognition.**

## Install env
```
conda create -n evmamba python=3.10.13
conda activate evmamba
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Please refer to [Vmamba](https://github.com/MzeroMiko/VMamba) to install the following packages：
```
cd kernels/selective_scan && pip install .
```

Install the required packages in mmaction
```
pip install -U openmim
mim install mmengine
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -v -e .
```

Download pre-trained [vssmbase_dp06_ckpt_epoch_241](https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmbase_dp06_ckpt_epoch_241.pth) and put it under `$/pretrained_models`

Then, put the HAR dataset Celex-HAR in `./data`

You can modify the paths of pretrained_model and dataset by editing
```
EVMamba\mmaction\models\backbones\classification\config_b.py
EVMamba\configs\recognition\EVMamba\CeleX-HAR.py
```

# :runner: Train & Test

```
# train & test
bash train.sh
bash dist_train.sh  (For Distributed Training)
```

# :open_book: Download the CeleX-HAR dataset 

* **BaiduYun**: 

:floppy_disk: **Baidu Netdisk** link: https://pan.baidu.com/s/1yvJje7DqIn8qT9fmQMbeeQ?pwd=wsad code：wsad


The directory should have the below format:
```Shell
├── CeleX-HAR (124,625 videos (training subset: 99,642 videos;  testing subset: 24,983 videos;))
    ├── CeleX_HR (377.32GB)
        ├── rawframes
            ├── action_001_pull_up_weeds
                ├── action_001_20220221_110904108_EI_70M
                    ├── 0000.png
                    ├── 0001.png
                    ├── 0002.png
                    ├── ...
                ├── action_001_20220221_110910254_EI_70M
                ├── ...
            ├── action_002_take_somebody's_pulse
            ├── action_003_move_the_chair
            ├── ...
    ├── celex_voxel (15.55GB)
        ├── action_001_pull_up_weeds
            ├── action_001_20220221_110904108_EI_70M.mat
            ├── action_001_20220221_110910254_EI_70M.mat
            ├── ...
        ├── action_002_take_somebody's_pulse
        ├── action_003_move_the_chair
        ├── ...
```


# :two_hearts: Citation 

If you have any questions about this work, please leave an issue. Also, please give us a **star** if you think this paper helps your research. 

```
@article{wang2024celexhar,
  title={Event Stream based Human Action Recognition: A High-Definition Benchmark Dataset and Algorithms},
  author={Wang, Xiao and Wang, Shiao and Shao, Pengpeng and Jiang, Bo and Zhu, Lin and Tian, Yonghong},
  journal={arXiv preprint arXiv:2408.09764},
  year={2024}
}
```



