#!/bin/bash

DATA_DIR="datasets3d/1_1_1_s176v2"
JSON_LIST="dataset_hecktor.json"
SLICE_BS=3
# Baseline Training of models on Hecktor and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on Hecktor

ckpt_paths=("Results_hecktor_normal_training/unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/swin_unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/segresnet/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_bot/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_enc/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/unet/data_hecktor/natural/model_latest.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")

echo "==========================================================================================="
echo "==========================================================================================="

echo "********************************Running Attack on Hecktor*********************************"

for ((i=0; i<6; i++)); do

   model_name=${model_names[$i]}
   ckpt_path=${ckpt_paths[$i]}

   echo "==========================================================================================="
   echo "Loading Model $model_name with Checkpoint $ckpt_path"

   # check if the checkpoint path exists
    if [ ! -f $ckpt_path ]; then
      echo "Checkpoint path $ckpt_path does not exist. Skipping..."
      continue
    fi


  echo "PGD Attack on Hecktor with PGD eps 4 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name pgd --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "PGD Attack on Hecktor with PGD eps 8 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name pgd --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "CosPGD Attack on Hecktor with  eps 4 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name cospgd --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "CosPGD Attack on Hecktor with  eps 8 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name cospgd --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"



  echo "PGD Attack on Hecktor with FGSM eps 4 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name fgsm --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "PGD Attack on Hecktor with FGSM eps 8 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name fgsm --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"



  echo "PGD Attack on Hecktor with GN std 4 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST  --slice_batch_size $SLICE_BS \
  --attack_name gn --eps 4  --std 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"

#
  echo "PGD Attack on Hecktor with GN std 8 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name gn --eps 8 --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "PGD Attack on Hecktor with VAFA-3D with qmax 10 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 4  --std 4 --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

  echo "PGD Attack on Hecktor with VAFA-3D with qmax 20 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 8 --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

  echo "PGD Attack on Hecktor with VAFA-3D with qmax 30 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 8 --std 8 --q_max 30 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

done


