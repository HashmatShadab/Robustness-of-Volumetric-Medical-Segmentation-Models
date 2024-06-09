#!/bin/bash

DATA_DIR="datasets3d/btcv-synapse"
JSON_LIST="dataset_synapse_18_12.json"
SLICE_BS=3
# Baseline Training of models on BTCV and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on BTCV

ckpt_paths=("Results/unetr/data_btcv/natural/model_latest.pt" "Results/swin_unetr/data_btcv/natural/model_latest.pt" "Results/segresnet/data_btcv/natural/model_latest.pt" "Results/umamba_bot/data_btcv/natural/model_latest.pt" "Results/umamba_enc/data_btcv/natural/model_latest.pt" "Results/unet/data_btcv/natural/model_latest.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")


# use echo to have a clear separation between different experiments

echo "==========================================================================================="
echo "==========================================================================================="


for ((j=0; j<6; j++)); do


  model_name=${model_names[$j]}
  ckpt_path=${ckpt_paths[$j]}



  # check if the checkpoint path exists
  if [ ! -f $ckpt_path ]; then
    echo "Checkpoint path $ckpt_path does not exist. Skipping..."
    continue
  fi

  for ((i=0; i<6; i++)); do

  # loop over model names

  if [ $i -eq $j ]; then
    echo "Skipping the same model $model_name"
    continue
  fi

  surrogate_model_name=${model_names[$i]}
  surrogate_ckpt_path=${ckpt_paths[$i]}

  surrogate_parent_dir=$(dirname $surrogate_ckpt_path)

#  echo "Surrogate Model $surrogate_model_name with parent directory $surrogate_parent_dir"

  # check if the checkpoint path exists
  if [ ! -f $surrogate_ckpt_path ]; then
    echo "Checkpoint path $surrogate_ckpt_path does not exist. Skipping..."
    continue
  fi

  # Loop over all the folders in parent directory starting with "Adv"
  for adv_dir in $surrogate_parent_dir/Adv*/btcv/*; do
    # run code only if the folder exists
    if [ -d "$adv_dir" ]; then

      echo "Evaluating $model_name on Adversarial Images in $adv_dir"
      python inference_on_adv_images.py --model_name $model_name --pos_embed conv --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset btcv --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
      --adv_imgs_dir $adv_dir

      echo "-------------------------------------------------------------------------------------------"
    fi

    done
  done
done

