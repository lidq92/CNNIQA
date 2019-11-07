#!/bin/bash
# nohup ./job.sh > CNNIQA-LIVE-0-10.log 2>&1 &
source activate reproducibleresearch
for ((i=0; i<10; i++)); do
    CUDA_VISIBLE_DEVICES=1 python main.py --exp_id=$i --database=LIVE
done;
source deactivate
