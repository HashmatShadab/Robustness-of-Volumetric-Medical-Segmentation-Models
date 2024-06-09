#!/bin/bash

DATA_DIR="E:\Projects_May2024\AdvTransferMed3D\datasets3d\btcv-synapse"
JSON_LIST="dataset_synapse_18_12.json"
SLICE_BS=3
# Baseline Training of models on BTCV and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on BTCV

#ckpt_paths=("Results/unetr/data_btcv/natural/model_latest.pt" "Results/swin_unetr/data_btcv/natural/model_latest.pt" "Results/segresnet/data_btcv/natural/model_latest.pt" "Results/umamba_bot/data_btcv/natural/model_latest.pt" "Results/umamba_enc/data_btcv/natural/model_latest.pt" "Results/unet/data_btcv/natural/model_latest.pt")
#model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")

ckpt_paths=("Results/unet/data_btcv/natural/model_latest.pt")
model_names=("unet")

echo "==========================================================================================="
echo "==========================================================================================="

echo "********************************Running Attack on BTCV*********************************"

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


  echo "PGD Attack on BTCV with PGD eps 4 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name pgd --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "PGD Attack on BTCV with PGD eps 8 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name pgd --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"



  echo "PGD Attack on BTCV with FGSM eps 4 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name fgsm --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "PGD Attack on BTCV with FGSM eps 8 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name fgsm --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"



  echo "PGD Attack on BTCV with GN std 4 $model_name"
  python wb_attack.py  --model_name $model_name  --pos_embed conv --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST  --slice_batch_size $SLICE_BS \
  --attack_name gn --eps 4  --std 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"

#
  echo "PGD Attack on BTCV with GN std 8 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name gn --eps 8 --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "PGD Attack on BTCV with VAFA-3D with qmax 10 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 4  --std 4 --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

  echo "PGD Attack on BTCV with VAFA-3D with qmax 20 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 8 --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

  echo "PGD Attack on BTCV with VAFA-3D with qmax 30 $model_name"
  python wb_attack.py  --model_name $model_name --pos_embed conv  --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 8 --std 8 --q_max 30 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

done

# use echo to have a clear separation between different experiments

echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Inference on BTCV*********************************"


for ((i=0; i<6; i++)); do

   model_name=${model_names[$i]}
   ckpt_path=${ckpt_paths[$i]}

   # get parent directory of the checkpoint path
   parent_dir=$(dirname $ckpt_path)

   echo "Inference on Adversarial Images on BTCV with model $model_name with Checkpoint $ckpt_path"
   # check if the checkpoint path exists
    if [ ! -f $ckpt_path ]; then
      echo "Checkpoint path $ckpt_path does not exist. Skipping..."
      continue
    fi

   # Loop over all the folders in parent directory starting with "Adv"
    for adv_dir in $parent_dir/Adv*/btcv/*; do
      # run code only if the folder exists
      if [ -d "$adv_dir" ]; then

          echo "Inference on Adversarial Images in $adv_dir"
        python inference_on_adv_images.py --model_name $model_name --pos_embed conv --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
        --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
        --adv_imgs_dir $adv_dir

        echo "-------------------------------------------------------------------------------------------"
      fi

    done



done





