#!/bin/bash


# Baseline Training of models on BTCV and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on BTCV

ckpt_paths=("Results/unetr/data_btcv/natural/model_latest.pt" "Results/swin_unetr/data_btcv/natural/model_latest.pt" "Results/segresnet/data_btcv/natural/model_latest.pt" "Results/umamba_bot/data_btcv/natural/model_latest.pt" "Results/umamba_enc/data_btcv/natural/model_latest.pt" "Results/unet/data_btcv/natural/model_latest.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")


# use echo to have a clear separation between different experiments

echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Inference on BTCV*********************************"


for ((i=0; i<6; i++)); do

   model_name=${model_names[$i]}
   ckpt_path=${ckpt_paths[$i]}

   # get parent directory of the checkpoint path
   parent_dir=$(dirname $ckpt_path)

    echo "======================================================================================================================"

   echo "Adversarial Images on BTCV with model $model_name with Checkpoint $ckpt_path"
   # check if the checkpoint path exists
    if [ ! -f $ckpt_path ]; then
      echo "Checkpoint path $ckpt_path does not exist. Skipping..."
      continue
    fi

    echo "Following Adversarial folders found in $parent_dir:"
    for adv_dir in $parent_dir/Adv*/btcv/cospgd*; do
      echo "$adv_dir"
    done

   # Loop over all the folders in parent directory starting with "Adv"
    for adv_dir in $parent_dir/Adv*/btcv/cospgd*; do
      # run code only if the folder exists
      if [ -d "$adv_dir" ]; then
          echo "Inference on Adversarial Folder in $adv_dir"
        python sam_med3d_data_creation.py --data_dir $adv_dir --new_data_dir SAM_BTCV --dataset btcv


      else
        echo "No adversarial images found in $parent_dir"
      fi

    done

    echo "======================================================================================================================"




done





