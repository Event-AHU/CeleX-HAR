NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/recognition/EVMamba/CeleX-HAR.py 2 \
--seed 0 --deterministic