#!/bin/bash


MAX_EPOCHS=5000
WARMUP_EPOCHS=50

DATA_DIR="datasets3d/acdc"
JSON_LIST="dataset_acdc_140_20_.json"
# Baseline Training of models on ACDC


echo "Running Natural Training on ACDC for UNET"
python training.py  --model_name unet --in_channels 1 --out_channel 4 --feature_size=16 \
--dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
--adv_training_mode False --freq_reg_mode False \
--attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
--max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_model_dir="./Results" \
--val_every 15

echo "Running Natural Training on ACDC for UNETR"
python training.py  --model_name unetr --in_channels 1 --out_channel 4 --feature_size=16 \
--dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
--adv_training_mode False --freq_reg_mode False \
--attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
--max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_model_dir="./Results" \
--val_every 15

echo "Running Natural Training on ACDC for SegResNet"
python training.py  --model_name segresnet --in_channels 1 --out_channel 4 --feature_size=16 \
--dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=3 \
--adv_training_mode False --freq_reg_mode False \
--attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
--max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_model_dir="./Results" \
--val_every 15


echo "Running Natural Training on ACDC for Swin-UNETR"
python training.py  --model_name swin_unetr --in_channels 1 --out_channel 4 --feature_size=16 \
--dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
--adv_training_mode False --freq_reg_mode False \
--attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
--max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_model_dir="./Results" \
--val_every 15



echo "Running Natural Training on ACDC for UMAMBA-ENC"
python training.py  --model_name umamba_enc --in_channels 1 --out_channel 4 --feature_size=16 \
--dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
--adv_training_mode False --freq_reg_mode False \
--attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
--max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_model_dir="./Results" \
--val_every 15




echo "Running Natural Training on ACDC for UMAMBA-BOT"
python training.py  --model_name umamba_bot --in_channels 1 --out_channel 4 --feature_size=16 \
--dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --batch_size=1 \
--adv_training_mode False --freq_reg_mode False \
--attack_name vafa-3d --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False --vafa_norm False \
--max_epochs $MAX_EPOCHS --warmup_epochs $WARMUP_EPOCHS --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_model_dir="./Results" \
--val_every 15
