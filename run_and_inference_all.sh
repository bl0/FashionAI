#!/bin/bash

class_names=('collar_design_labels' 'neckline_design_labels' 'skirt_length_labels' 'sleeve_length_labels' 'neck_design_labels' 'coat_length_labels' 'lapel_design_labels' 'pant_length_labels')

for i in $(seq 0 0); do

# train
# python pytorch_main.py --arch resnet50  --epochs 100 --gpus 2 --cur_class_idx $i --pretrain | tee train_log_$i.txt

# inference
python pytorch_main.py --arch resnet50 --gpus 1 --inference --resume ./models/best_models/resnet50_${class_names[i]}.pth.tar --cur_class_idx $i 

done

# cat all result to single file
cat results/* > submit/0328b.csv
