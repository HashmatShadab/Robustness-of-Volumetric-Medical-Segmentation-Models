#!/bin/bash


exp_num=$1
model_name=$2
batch_size=${3:-3}

MAX_EPOCHS=5000
WARMUP_EPOCHS=50

DATA_DIR="datasets3d/acdc"
JSON_LIST="dataset_acdc_140_20_.json"
# Baseline Training of models on ACDC and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on ACDC

if [ $exp_num -eq 1 ]
then
  echo "Running Natural Training on ACDC"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10

fi

if [ $exp_num -eq 2 ]
then
  echo "Running Adv Training on ACDC with PGD-K eps 4"
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name pgd --eps 4 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi


if [ $exp_num -eq 3 ]
then
  echo "Running Adv Training on ACDC with PGD-K eps 8"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name pgd --eps 8 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi


if [ $exp_num -eq 4 ]
then
  echo "Running Adv Training on ACDC with FGSM eps 4"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name fgsm --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi


if [ $exp_num -eq 5 ]
then
  echo "Running Adv Training on ACDC with FGSM eps 8"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name fgsm --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi


if [ $exp_num -eq 6 ]
then
  echo "Running Adv Training on ACDC with GN eps 4"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name gn --std 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi


if [ $exp_num -eq 7 ]
then
  echo "Running Adv Training on ACDC with GN eps 8"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name gn --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi

if [ $exp_num -eq 8 ]
then
  echo "Running Adv Training on ACDC with VAFA-3D qmax 20 block_size 32"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss True --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi


if [ $exp_num -eq 9 ]
then
  echo "Running Adv Training on ACDC with VAFA-3D qmax 30 block_size 32"
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 30 --steps 20 --block_size 32 32 32 --use_ssim_loss True --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi

if [ $exp_num -eq 10 ]
then
  echo "Running Adv Training on ACDC with VAFA-3D qmax 10 block_size 32"
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 4 --feature_size=16 \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss True --vafa_norm False \
    --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 10
fi