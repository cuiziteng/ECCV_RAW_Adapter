# RAW-Adapter Object Detection (LOD dataset, PASCAL RAW dataset)

### 📖 1: Dataset Download

**LOD Dataset** (low-light RAW detection dataset):

Download LOD dataset  from [Google Drive](https://drive.google.com/file/d/1Jkm4mvynWxc7lXSH3H9sLI0wJ6p6ftvZ/view?usp=sharing) or [百度网盘 (passwd: kf43)](https://pan.baidu.com/s/1FA9lw1WXk2dJ0jtlLeho5w), or find the LOD dataset [original provide link](https://github.com/ying-fu/LODDataset).

Unzip and place it in $./data$ under this folder, which format as:

```
--  data
     -- LOD_BMVC21
         -- RAW_dark (RAW data, "demosacing" in our paper)
         -- RGB_dark (default ISP RGB data)
         -- RAW_dark_InverseISP (InvISP processed RAW data, [CVPR 2021])
         -- RAW_dark_ECCV16_Micheal (ECCV16 ISP processed RAW data, [ECCV 2016])
         -- RAW-dark-Annotations (detection label)      
         -- trainval
```

**PASCAL RAW Dataset** (RAW detection dataset, normal-light/ low-light/ over-exposure):

Download LOD dataset from [Google Drive](https://drive.google.com/file/d/1686W89ALVvtfUvK8NMvqWaUCTLBqhW-p/view?usp=sharing) or [百度网盘 (passwd: kjv9)](https://pan.baidu.com/s/1O76R8ZFZdLw88N0b3hT2Tw).

Unzip and place it in $./data$ under this folder, which format as:

```
--  data
     -- PASCAL_RAW_github
         -- annotations
         -- original (original RAW, demosaic RAW normal-light & over-exposure & low-light)
         -- compare_ISP (ISP methods, InvISP, ECCV16-ISP)
         -- trainval
```

**Note:** the original RAW file in PASCAL RAW are too big (>100GB), you could download them from [PASCAL RAW webiste](https://purl.stanford.edu/hq050zr7488), the code translate original RAW data to demosaic RAW data (normal-light, over-exposure, low-light) could find in here: [PASCAL_RAW_pre_process.py](PASCAL_RAW_pre_process.py).


### 📖 2: Enviroment Setup

Our code based on [mmdetection 3.3.0](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file) version, you can following their [instruction](https://mmdetection.readthedocs.io/en/latest/get_started.html) to build environment. Or following our steps below:

(1). Create a conda environment and activate it:
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

(2). Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g. ours (torch1.12.1+cu113):
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

(3). Mmdetection setup:
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

Develop mmdet and install rawpy:
```
pip install -v -e .
pip install rawpy
```

### 📖 3: Model Evaluation

Check the whole pretrain weights at [release](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases).

#### 3.1: **LOD Dataset** config & pretrain weights (ckpt).

Compare methods:

(1). Demosaic (RAW): Model trained on RAW data in LOD dataset

(2). Default ISP: Model trained on sRGB data in LOD dataset

(3). ECCV16-ISP: Model trained on sRGB data translated from [ECCV16-ISP](https://karaimer.github.io/camera-pipeline/)

(4). InvISP: Model trained on sRGB data translated from [InvISP](https://yzxing87.github.io/InvISP/index.html)

RetinaNet - ResNet50 backbone: 

| RAW-Adapter | Demosaic (RAW) | Default ISP | ECCV16-ISP | InvISP | 
|  ---- |  ---- | ---- | ---- | ----  | 
|  62.1  / [config](configs/LOD/R_Net_raw_adapter.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/R_Net_RAW_Adapter.pth) |  58.5  / [config](configs/LOD/R_Net_demosaic.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/R_Net_demosaic.pth) | 58.4  / [config](configs/LOD/R_Net_default_ISP.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/R_Net_default_ISP.pth) | 54.4  / [config](configs/LOD/R_Net_ECCV_16.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/R_Net_ECCV_16.pth) | 56.9  / [config](configs/LOD/R_Net_InvISP.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/R_Net_InvISP.pth) |

SP-RCNN - ResNet50 backbone: 

| RAW-Adapter | Demosaic (RAW) | Default ISP | ECCV16-ISP | InvISP | 
|  ---- |  ---- | ---- | ---- | ----  | 
|  59.2  / [config](configs/LOD/SpRCNN_raw_adapter.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/SP_RCNN_RAW_Adapter.pth) |  57.7  / [config](configs/LOD/SpRCNN_demosaic.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/SP_RCNN_demosaic.pth) | 53.9  / [config](configs/LOD/SpRCNN_default_ISP.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/SP_RCNN_default_ISP.pth) | 52.2  / [config](configs/LOD/SpRCNN_ECCV16.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/SP_RCNN_ECCV_16.pth) | 49.4  / [config](configs/LOD/SpRCNN_InvISP.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/SP_RCNN_InvISP.pth) |

Evaluation of RAW-Adapter or comparision methods, only need single GPU (RAW-Adapter, RetinaNet for example), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/LOD/R_Net_raw_adapter.py https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.1/R_Net_RAW_Adapter.pth
```

#### 3.2: **PASCAL RAW Dataset** config & pretrain weights (ckpt):

RetinaNet - ResNet18 backbone: 

|  Light Conditions | **RAW-Adapter** | Demosaic (RAW)  | ECCV16-ISP | InvISP | 
|  ---- |  ---- |  ---- | ---- | ---- | 
|  Normal-Light | 88.7  / [config](configs/PASCALRAW_Res18/Normal_Light_raw_adapter_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_RAW_Adapter_res18.pth)  |  87.7  / [config](configs/PASCALRAW_Res18/Normal_Light_demosaic_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_Demosaic_res18.pth) | 88.1  / [config](configs/PASCALRAW_Res18/Normal_Light_eccv16_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_ECCV_16_res18.pth) | 85.4  / [config](configs/PASCALRAW_Res18/Normal_Light_inverse_isp_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_InvISP_res18.pth) |
|  Over-Exposure |  88.7  / [config](configs/PASCALRAW_Res18/Over_Exp_raw_adapter_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_RAW_Adapter_res18.pth) | 87.7 / [config](configs/PASCALRAW_Res18/Over_Exp_demosaic_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_Demosaic_res18.pth)  | 85.6 / [config](configs/PASCALRAW_Res18/Over_Exp_eccv16_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_ECCV_16_res18.pth) | 86.6 / [config](configs/PASCALRAW_Res18/Over_Exp_inverse_isp_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_InvISP_res18.pth) |
|  Low-Light | 82.5  / [config](configs/PASCALRAW_Res18/Low_Light_raw_adapter_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_RAW_Adapter_res18.pth)  | 80.3  / [config](configs/PASCALRAW_Res18/Low_Light_demosaic_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_Demosaic_res18.pth)  | 78.8  / [config](configs/PASCALRAW_Res18/Low_Light_eccv16_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_ECCV_16_res18.pth) | 70.9  / [config](configs/PASCALRAW_Res18/Low_Light_inverse_isp_res18.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_InvISP_res18.pth) |

RetinaNet - ResNet50 backbone: 

|  Light Conditions | RAW-Adapter | Demosaic (RAW)  | ECCV16-ISP | InvISP | 
|  ---- |  ---- |  ---- | ---- | ---- | 
|  Normal-Light | 89.7  / [config](configs/PASCALRAW_Res50/Normal_Light_raw_adapter_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_RAW_Adapter_res50.pth)  |  89.2  / [config](configs/PASCALRAW_Res50/Normal_Light_demosaic_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_Demosaic_res50.pth) | 89.4  / [config](configs/PASCALRAW_Res50/Normal_Light_eccv16_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_ECCV_16_res50.pth) | 87.6  / [config](configs/PASCALRAW_Res50/Normal_Light_inverse_isp_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/Normal_InvISP_res50.pth) |
|  Over-Exposure |  89.5  / [config](configs/PASCALRAW_Res50/Over_Exp_raw_adapter_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_RAW_Adapter_res50.pth) | 88.8 / [config](configs/PASCALRAW_Res50/Over_Exp_demosaic_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_Demosaic_res50.pth)  | 86.8 / [config](configs/PASCALRAW_Res50/Over_Exp_eccv16_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_ECCV_16_res50.pth) | 87.3 / [config](configs/PASCALRAW_Res50/Over_Exp_inverse_isp_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/OE_InvISP_res50.pth) |
|  Low-Light | 86.6  / [config](configs/PASCALRAW_Res50/Low_Light_raw_adapter_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_RAW_Adapter_res50.pth)  | 82.6  / [config](configs/PASCALRAW_Res50/Low_Light_demosaic_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_Demosaic_res50.pth)  | 79.6  / [config](configs/PASCALRAW_Res50/Low_Light_eccv16_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_ECCV_16_res50.pth) | 74.7  / [config](configs/PASCALRAW_Res50/Low_Light_inverse_isp_res50.py)  [[ckpt]](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_InvISP_res50.pth) |

Evaluation of RAW-Adapter or comparision methods, only need single GPU (RAW-Adapter, ResNet18, Low-light for example), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/PASCALRAW_Res18/Low_Light_raw_adapter_res18.py https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.2/LL_RAW_Adapter_res18.pth
```

### 📖 4: Model Training (Optional)

We default train all RetinaNet model on 1 GPU: (PASCAL RAW dataset, RAW-Adapter (ResNet-18) for example)

```
python tools/train.py configs/PASCALRAW_Res18/Normal_Light_raw_adapter_res18.py
```

We default train all SP-RCNN model on 4 GPUs: (LOD dataset, RAW-Adapter for example)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29588 bash tools/dist_train.sh configs/LOD/SpRCNN_raw_adapter.py
```


### 📖 5: Additional  Information

If you want to editing the code or find out details of RAW-Adapter, direct refer to [mmdet/models/backbones/RAW_Adapter](mmdet/models/backbones/RAW_Adapter) and [mmdet/models/backbones/RAW_resnet.py](mmdet/models/backbones/RAW_resnet.py).


### 📖 Acknowledgement:

We thanks mmdetection & LOD & PASCAL RAW for their excellent code base & dataset.
