#!/bin/bash


# Baseline Training of models on Hecktor and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on Hecktor

ckpt_paths=("Results/unetr/data_abdomen/natural/model_latest.pt" "Results/swin_unetr/data_abdomen/natural/model_latest.pt" "Results/segresnet/data_abdomen/natural/model_latest.pt" "Results/umamba_bot/data_abdomen/natural/model_latest.pt" " Results_no_amp/umamba_enc/data_abdomen/natural/model_latest.pt"  "Results/unet/data_abdomen/natural/model_latest.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")

# use echo to have a clear separation between different experiments

echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Attack on Abdomen CT*********************************"



for ((i=0; i<6; i++)); do

   model_name=${model_names[$i]}
   ckpt_path=${ckpt_paths[$i]}

   # get parent directory of the checkpoint path
   parent_dir=$(dirname $ckpt_path)

    echo "======================================================================================================================"

   echo "Adversarial Images on Hecktor with model $model_name with Checkpoint $ckpt_path"
   # check if the checkpoint path exists
    if [ ! -f $ckpt_path ]; then
      echo "Checkpoint path $ckpt_path does not exist. Skipping..."
      continue
    fi

    echo "Following Adversarial folders found in $parent_dir:"
    for adv_dir in $parent_dir/Adv*/abdomen/cospgd*; do
      echo "$adv_dir"
    done

   # Loop over all the folders in parent directory starting with "Adv"
    for adv_dir in $parent_dir/Adv*/abdomen/cospgd*; do
      # run code only if the folder exists
      if [ -d "$adv_dir" ]; then
          echo "Inference on Adversarial Folder in $adv_dir"
        python sam_med3d_data_creation.py --data_dir $adv_dir --new_data_dir SAM_Abdomen --dataset abdomen


      else
        echo "No adversarial images found in $parent_dir"
      fi

    done

    echo "======================================================================================================================"




done





