#!/bin/bash

##### test ####
python ./test.py --config_file='configs/softmax_triplet_veri.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/home/williamramirez/Desktop/codigoswilliam/results/PGAN/VeRi/baseline/model_best.pth')" \
OUTPUT_DIR "('./')"






