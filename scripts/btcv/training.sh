#!/bin/bash


exp_num=$1
model_name=$2
batch_size=${3:-3}

DATA_DIR="datasets3d/btcv-synapse"
JSON_LIST="dataset_synapse_18_12.json"
# Baseline Training of models on BTCV and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on BTCV
# unetr models for btcv are trained with pos_embed to conv

if [ $exp_num -eq 1 ]
then
  echo "Running Natural Training on BTCV"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15

fi

if [ $exp_num -eq 2 ]
then
  echo "Running Adv Training on BTCV with PGD-K eps 4"
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name pgd --eps 4 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 3 ]
then
  echo "Running Adv Training on BTCV with PGD-K eps 8"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name pgd --eps 8 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 4 ]
then
  echo "Running Adv Training on BTCV with FGSM eps 4"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name fgsm --eps 4 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 5 ]
then
  echo "Running Adv Training on BTCV with FGSM eps 8"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name fgsm --eps 8 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 6 ]
then
  echo "Running Adv Training on BTCV with GN eps 4"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name gn --std 4 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 7 ]
then
  echo "Running Adv Training on BTCV with GN eps 8"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name gn --std 8 --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi

if [ $exp_num -eq 8 ]
then
  echo "Running Adv Training on BTCV with VAFA-3D qmax 20 block_size 32"
    python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16  \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 20 --steps 10 --block_size 32 32 32 --use_ssim_loss True \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 9 ]
then
  echo "Running Adv Training on BTCV with VAFA-3D qmax 30 block_size 32"
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16  \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 30 --steps 10 --block_size 32 32 32 --use_ssim_loss True \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi

if [ $exp_num -eq 10 ]
then
  echo "Running Adv Training on BTCV with VAFA-3D qmax 10 block_size 32"
     python adv_training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 10 --steps 10 --block_size 32 32 32 --use_ssim_loss True \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi
