#!/bin/bash


MAX_EPOCHS=500
WARMUP_EPOCHS=50

DATA_DIR="datasets3d/1_1_1_s176v2"
JSON_LIST="dataset_hecktor.json"
# Baseline Training of models on Hecktor and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on Hecktor

echo "Running Natural Training on Hecktor unetr"
  python training.py  --model_name unetr --in_channels 1 --out_channel 3 --feature_size=16 \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
  --adv_training_mode False --freq_reg_mode False \
  --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
  --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
  --save_model_dir="./Results" \
  --val_every 10

echo "Running Natural Training on Hecktor segresnet"
  python training.py  --model_name segresnet --in_channels 1 --out_channel 3 --feature_size=16 \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
  --adv_training_mode False --freq_reg_mode False \
  --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
  --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
  --save_model_dir="./Results" \
  --val_every 10

echo "Running Natural Training on Hecktor swin_unetr"
  python training.py  --model_name swin_unetr --in_channels 1 --out_channel 3 --feature_size=16 \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
  --adv_training_mode False --freq_reg_mode False \
  --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
  --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
  --save_model_dir="./Results" \
  --val_every 10

echo "Running Natural Training on Hecktor umamba_bot"
  python training.py  --model_name umamba_bot --in_channels 1 --out_channel 3 --feature_size=16 \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
  --adv_training_mode False --freq_reg_mode False \
  --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
  --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
  --save_model_dir="./Results" \
  --val_every 10


echo "Running Natural Training on Hecktor umamba_enc"
  python training.py  --model_name umamba_enc --in_channels 1 --out_channel 3 --feature_size=16 \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
  --adv_training_mode False --freq_reg_mode False \
  --attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \
  --max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
  --save_model_dir="./Results" \
  --val_every 10
