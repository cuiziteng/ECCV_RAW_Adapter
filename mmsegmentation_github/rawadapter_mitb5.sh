# Segformer with MIT B5 backbone
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29587 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_normal.py 4 
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29587 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_low.py 4 
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29587 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_oe.py 4 
