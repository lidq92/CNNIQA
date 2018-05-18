#!/bin/bash
# nohup ./jobCNNIQA.sh > CNNIQA-LIVE-0-10.log 2>&1 &
source activate ~/anaconda3/envs/tensorflow/
for ((i=0; i<10; i++)); do
    CUDA_VISIBLE_DEVICES=1 python CNNIQA.py $i config.yaml LIVE CNNIQA
done;
source deactivate