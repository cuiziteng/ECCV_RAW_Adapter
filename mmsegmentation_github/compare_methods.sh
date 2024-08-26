## Compare Methods (Inverse ISP, SID, ECCV 16 ISP)
## We default use 4 GPUs to run all the experiments, if you adjust GPU number, don't forget to change learning rate ~~
## All Segformer with MIT-B5 Backbone

# Inverse ISP
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/cvpr21inverseisp_low.py 4 --work-dir work_dirs/RAW_Adapter/inverseisp/low
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/cvpr21inverseisp_normal.py 4 --work-dir work_dirs/RAW_Adapter/inverseisp/normal
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/cvpr21inverseisp_oe.py 4 --work-dir work_dirs/RAW_Adapter/inverseisp/oe

# SID
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/cvpr18sid_low.py 4 --work-dir work_dirs/RAW_Adapter/SID/low

# ECCV16 ISP
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv16isp_low.py 4 --work-dir work_dirs/RAW_Adapter/eccv16isp/low
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv16isp_normal.py 4 --work-dir work_dirs/RAW_Adapter/eccv16isp/normal
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv16isp_oe.py 4 --work-dir work_dirs/RAW_Adapter/eccv16isp/oe
