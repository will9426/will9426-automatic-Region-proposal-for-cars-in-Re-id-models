#!/bin/bash

##### test ####
python ./test.py --config_file='configs/softmax_triplet_vric.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/home/williamramirez/Desktop/codigoswilliam/results/PGAN/VRIC/ourmodel_large_data/resnet50_model_210.pth')" \
OUTPUT_DIR "('./')"






