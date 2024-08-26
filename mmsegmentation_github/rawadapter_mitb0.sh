# Segformer with MIT B0 backbone
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29587 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_normal_mitb0.py 4 
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29587 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_low_mitb0.py 4 
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29587 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_oe_mitb0.py 4 
