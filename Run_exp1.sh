conda activate DRPC_REID
cd custom_model
mkdir weigths
cd weigths
gdown --id "1UWjwzsfXdTiHOH6elFXf4-knDT07qZbY" -O "exp01.pth"
cd ../../
python ./test.py --config_file='configs/softmax_triplet_vric1.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('./custom_model/weigths/exp01.pth')" \
OUTPUT_DIR "('./')"




