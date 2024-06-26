#!/bin/bash


exp_num=$1
model_name=$2
batch_size=${3:-3}

DATA_DIR="datasets3d/abdomenCT_resampled"
JSON_LIST="dataset.json"
# Baseline Training of models on Abdomen-CT and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on Abdomen-CT


if [ $exp_num -eq 1 ]
then
  echo "Running Natural Training on Abdomen-CT"
    python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15

fi

if [ $exp_num -eq 2 ]
then
  echo "Running Adv Training on Abdomen-CT with PGD-K eps 4"
     python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name pgd --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 3 ]
then
  echo "Running Adv Training on Abdomen-CT with PGD-K eps 8"
    python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name pgd --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 4 ]
then
  echo "Running Adv Training on Abdomen-CT with FGSM eps 4"
    python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name fgsm --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 5 ]
then
  echo "Running Adv Training on Abdomen-CT with FGSM eps 8"
    python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name fgsm --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 6 ]
then
  echo "Running Adv Training on Abdomen-CT with GN eps 4"
    python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name gn --std 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 7 ]
then
  echo "Running Adv Training on Abdomen-CT with GN eps 8"
    python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode False \
    --attack_name gn --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi

if [ $exp_num -eq 8 ]
then
  echo "Running Adv Training on Abdomen-CT with VAFA-3D qmax 20 block_size 32"
    python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 --pos_embed conv \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss True \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi


if [ $exp_num -eq 9 ]
then
  echo "Running Adv Training on Abdomen-CT with VAFA-3D qmax 30 block_size 32"
     python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 --pos_embed conv \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 30 --steps 20 --block_size 32 32 32 --use_ssim_loss True \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi

if [ $exp_num -eq 10 ]
then
  echo "Running Adv Training on Abdomen-CT with VAFA-3D qmax 10 block_size 32"
     python training.py  --model_name $model_name --in_channels 1 --out_channel 14 --feature_size=16 --pos_embed conv \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=$batch_size \
    --adv_training_mode True --freq_reg_mode True \
    --attack_name vafa-3d  --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss True \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15
fi
