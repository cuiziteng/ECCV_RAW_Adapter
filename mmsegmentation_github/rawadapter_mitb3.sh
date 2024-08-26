# Segformer with MIT B3 backbone
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29666 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_normal_mitb3.py 4 
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29666 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_low_mitb3.py 4 
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29666 bash tools/dist_train.sh configs/ECCV_RAW_Adapter/eccv24_rawadapter_oe_mitb3.py 4 
