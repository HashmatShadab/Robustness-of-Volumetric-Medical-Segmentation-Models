#!/bin/bash


exp_num=$1
model_name=$2
batch_size=${3:-3}

WARMUP_EPOCHS=50

DATA_DIR="datasets3d/acdc"
JSON_LIST="dataset_acdc_140_20_.json"
# Baseline Training of models on ACDC and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on ACDC


if [ $exp_num -eq 1 ]
then
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs 5000 --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Check_acdc_5000" \
    --val_every 10

fi




if [ $exp_num -eq 2 ]
then
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss True --vafa_norm False \
    --max_epochs 5000 --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Check_acdc_5000" \
    --val_every 10
fi

if [ $exp_num -eq 3 ]
then
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs 5000 --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Check_acdc_5000" \
    --val_every 10
fi



