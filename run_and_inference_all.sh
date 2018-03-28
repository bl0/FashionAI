#!/bin/bash

class_names=('collar_design_labels' 'neckline_design_labels' 'skirt_length_labels' 'sleeve_length_labels' 'neck_design_labels' 'coat_length_labels' 'lapel_design_labels' 'pant_length_labels')

arch=resnet50
gpu=$1

for i in $(seq 2 2); do {

# train
python pytorch_main.py \
    --arch ${arch} \
    --epochs 30 \
    --gpus i \
    --cur_class_idx $i \
    -b 32 \
    --pretrain  | tee logs/train_${arch}_log_$i.txt

# inference
python pytorch_main.py \
    --arch ${arch} \
    --gpus $i \
    --inference \
    --resume ./models/best_models/${arch}_${class_names[i]}.pth.tar \
    --cur_class_idx $i \
    --save_path ./results/${arch}
} &
done

# cat all result to single file
# cat results/* > submit/0328b_${arch}.csv
