#!/bin/bash


exp_num=$1

DATA_DIR="datasets3d/abdomenCT_resampled"
JSON_LIST="dataset.json"
# Baseline Training of models on Abdomen-CT and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on Abdomen-CT


if [ $exp_num -eq 1 ]
then
  echo "Running Natural Training on Abdomen-CT"
    python training.py  --model_name unet --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15

    echo "Running Natural Training on Abdomen-CT"
    python training.py  --model_name unetr --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15

    echo "Running Natural Training on Abdomen-CT"
    python training.py  --model_name segresnet --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15

fi


if [ $exp_num -eq 2 ]
then
  echo "Running Natural Training on Abdomen-CT"
    python training.py  --model_name swin_unetr --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15

    echo "Running Natural Training on Abdomen-CT"
    python training.py  --model_name umamba_bot --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results" \
    --val_every 15


fi


if [ $exp_num -eq 3 ]
then
  echo "Running Natural Training on Abdomen-CT"
    python training.py --noamp True  --model_name umamba_enc --in_channels 1 --out_channel 14 --feature_size=16 \
    --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
    --adv_training_mode False --freq_reg_mode False \
    --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
    --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
    --save_model_dir="./Results_no_amp" \
    --val_every 15



fi

