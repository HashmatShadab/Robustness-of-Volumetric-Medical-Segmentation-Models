#!/bin/bash

DATA_DIR="datasets3d/hecktor"
JSON_LIST="dataset_hecktor.json"
SLICE_BS=3
# Baseline Training of models on BTCV CT and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on BTCV CT

ckpt_paths=("Results_hecktor_normal_training/unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/swin_unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/segresnet/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_bot/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_enc/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/unet/data_hecktor/natural/model_latest.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")


# use echo to have a clear separation between different experiments

echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Inference on Hecktor CT*********************************"



for ((j=0; j<6; j++)); do


  model_name=${model_names[$j]}
  ckpt_path=${ckpt_paths[$j]}

  # check if the checkpoint path exists
  if [ ! -f $ckpt_path ]; then
    echo "Checkpoint path $ckpt_path does not exist. Skipping..."
    continue
  fi

  model_parent_dir=$(dirname $ckpt_path)
  echo "Model $model_name with parent directory $model_parent_dir"




  # Loop over all the folders in parent directory starting with "Adv"
  for adv_dir in $model_parent_dir/Adv*/hecktor/*; do
    # run code only if the folder exists
    if [ -d "$adv_dir" ]; then

      echo "Evaluating $model_name on Adversarial Images in $adv_dir"
      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 8

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 16

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 32

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 2 --upper_limit 64

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 8 --upper_limit 96

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 16 --upper_limit 96

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 32 --upper_limit 96


      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 16 --upper_limit 48


      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 32 --upper_limit 64


      echo "-------------------------------------------------------------------------------------------"
    fi

    done




done

