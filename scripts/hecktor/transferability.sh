#!/bin/bash

DATA_DIR="datasets3d/hecktor"
JSON_LIST="dataset_hecktor.json"
SLICE_BS=3


ckpt_paths=("Results_hecktor_normal_training/unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/swin_unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/segresnet/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_bot/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_enc/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/unet/data_hecktor/natural/model_latest.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")

echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Inference on Hecktor*********************************"


for ((i=0; i<6; i++)); do

   model_name=${model_names[$i]}
   ckpt_path=${ckpt_paths[$i]}

   # get parent directory of the checkpoint path
   parent_dir=$(dirname $ckpt_path)

   echo "Inference on Adversarial Images on Hecktor with model $model_name with Checkpoint $ckpt_path"
   # check if the checkpoint path exists
    if [ ! -f $ckpt_path ]; then
      echo "Checkpoint path $ckpt_path does not exist. Skipping..."
      continue
    fi

   # Loop over all the folders in parent directory starting with "Adv"
    for adv_dir in $parent_dir/Adv*/hecktor/*; do
      # run code only if the folder exists
      if [ -d "$adv_dir" ]; then

          echo "Inference on Adversarial Images in $adv_dir"
        python inference_on_adv_images.py --model_name $model_name --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
        --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
        --adv_imgs_dir $adv_dir

        echo "-------------------------------------------------------------------------------------------"
      fi

    done



done



