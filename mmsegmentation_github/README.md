# RAW-Adapter Semantic Segmentation (ADE20K-RAW dataset)

### 1: Dataset Download

**ADE20K-RAW** is our synthesised RAW data semantic segmentation dataset, we adopt InverseISP to translate RGB to raw-RGB data, then apply inverse white balance & mosaic on the translated data.

Download the dataset from [Google Drive](https://drive.google.com/file/d/1OZ4_rbJqlmlvmIjWzM5J4JjQCF2-fatP/view?usp=sharing) or [百度网盘 (passwd: acv7)](https://pan.baidu.com/s/1hv4Dc6AGBRr1u-7OgJ0zfA)

Dataset Format as:

```
--  data
     -- ADE20K
         -- ADEChallengeData2016
             
             -- annotations
                 -- training
                 -- validation
             
             -- images

                 # Synthesis RAW data
                 -- training_raw (synthesis RAW)
                 -- validation_raw
                 -- training_raw_low  (synthesis RAW, low-light)
                 -- validation_raw_low
                 -- training_raw_over_exp  (synthesis RAW, over-exposure)
                 -- validation_raw_over_exp

                 # Compare ISP methods
                  -- ISP_methods
                      -- 1_SID (https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf)
                      -- 2_InvISP (https://yzxing87.github.io/InvISP/index.html)
                      -- 3_ECCV16(https://karaimer.github.io/camera-pipeline/)
```

### 2: Enviroment Setup

Our code based on [mmsegmentation 1.2.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.1) version, you can following their [instruction](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.1/docs/en/get_started.md#installation) to build environment. Or following our steps below:

(1). Create a conda environment and activate it:
```
conda create --name mmseg python=3.8 -y
conda activate mmseg
```

(2). Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g. ours (torch1.11.0+cu113):
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

(3). Mmsegmentation setup:
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

Develop mmseg:
```
pip install -v -e .
```


### 3: Model Evaluation 


We show the benchmark and **mIOU** performance. Download the pretrain weights (ckpt) and training logs (log) from:

**RAW-Adapter** (Segformer - MITB5 & B3 & B0 backbone): 

| Light Condition | Normal-Light | Over-Exposure | Low-Light | 
|  ---- | ---- | ---- | ----   | 
| RAW-Adapter (MITB5) | 47.95 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_normal.py)  [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_normal_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_normal.log)] |  46.62 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_oe.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_oe_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_oe.log)] | 38.75 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_low.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_low_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_low.log)] | 
| RAW-Adapter (MITB3) | 46.57 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_normal_mitb3.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_normal_mitb3_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_normal_mitb3.log)] | 44.19 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_oe_mitb3.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_oe_mitb3_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_oe_mitb3.log)] | 37.62 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_low_mitb3.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_low_mitb3_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_low_mitb3.log)] | 
| RAW-Adapter (MITB0) | 34.72 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_normal_mitb0.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_normal_mitb0_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_normal_mitb0.log)] | 31.91 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_oe_mitb0.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_oe_mitb0_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_oe_mitb0.log)] | 25.06 / [config](configs/ECCV_RAW_Adapter/eccv24_rawadapter_low_mitb0.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_low_mitb0_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/raw_adapter_low_mitb0.log)] | 


Evaluation of RAW-Adapter, only need single GPU (low-light, MIT-B5 backbone for example), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/ECCV_RAW_Adapter/eccv24_rawadapter_low.py raw_adapter_low_iter_80000.pth(your weight path) 
```

**Compare Methods:** Demosaic (RAW), [ECCV16-ISP](https://karaimer.github.io/camera-pipeline/), [InvISP](https://yzxing87.github.io/InvISP/index.html), [SID](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf) (Segformer - MITB5 backbone): 

| Light Condition | Normal-Light | Over-Exposure | Low-Light | 
|  ---- | ---- | ---- | ----   | 
| Demosaic (MITB5) | 47.47  / [config](configs/ECCV_RAW_Adapter/demosaic_normal.py)  [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/Demosaic_normal_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/Demosaic_normal.log)] | 45.69 / [config](configs/ECCV_RAW_Adapter/demosaic_oe.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/Demosaic_oe_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/Demosaic_oe.log)] | 37.55 / [config](configs/ECCV_RAW_Adapter/demosaic_low.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_low_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_low.log)] | 
| ECCV16-ISP (MITB5) | 45.48 / [config](configs/ECCV_RAW_Adapter/eccv16isp_normal.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_normal_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_normal.log)] | 42.85 / [config](configs/ECCV_RAW_Adapter/eccv16isp_oe.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_oe_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_oe.log)] | 37.32 / [config](configs/ECCV_RAW_Adapter/eccv16isp_low.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_low_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/eccv16isp_low.log)] | 
| InvISP (MITB5) | 47.82 / [config](configs/ECCV_RAW_Adapter/cvpr21inverseisp_normal.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/inverseisp_normal_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/inverseisp_normal.log)] | 44.30 / [config](configs/ECCV_RAW_Adapter/cvpr21inverseisp_oe.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/inverseisp_oe_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/inverseisp_oe.log)] |  4.03  / [config](configs/ECCV_RAW_Adapter/cvpr21inverseisp_low.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/inverseisp_low_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/inverseisp_low.log)] |
| SID (MITB5) |  ---- |  ---- | 37.06 / [config](configs/ECCV_RAW_Adapter/cvpr18sid_low.py) [[ckpt](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/SID_low_iter_80000.pth), [log](https://github.com/cuiziteng/ECCV_RAW_Adapter/releases/download/1.0.0/SID_low.log)] |

Evaluation of comparision methods, only need single GPU (low-light, SID for example), if you need visulization, please add "--show-dir": 

```
python tools/test.py configs/ECCV_RAW_Adapter/cvpr18sid_low.py SID_low_iter_80000.pth(your weight path) 
```

### 4: Model Training (Optional) 

We default train all the model on 4 GPUs, with the same data augmentation, resolution, learning rate and iters (80000), etc.
If you train model with other GPU number or batch size, remember to adjust the learning rate. 

(1). Training code of RAW-Adapter (low-light, MIT-B5 backbone for example):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29588 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_low.py 4
```

Or you can direct train with bash (RAW-Adapter with MIT-B5/ B3/ B0 backbone):

```
bash rawadapter_mitb0.sh
bash rawadapter_mitb3.sh
bash rawadapter_mitb5.sh
```

(2). Training code of comparision methods (low-light, SID method for example):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29588 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/cvpr18sid_low.py 4
```


### 5: Additional  Information

If you want to editing the code or find out details of RAW-Adapter, direct refer to [mmseg/models/backbones/RAW_Adapter](mmseg/models/backbones/RAW_Adapter) and [mmseg/models/backbones/raw_mit.py](mmseg/models/backbones/raw_mit.py).


### Acknowledgement:

We thanks [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.1) & [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) for their excellent code base & dataset, and [InvISP](https://yzxing87.github.io/InvISP/index.html) for the RAW data synthesis contribution.





