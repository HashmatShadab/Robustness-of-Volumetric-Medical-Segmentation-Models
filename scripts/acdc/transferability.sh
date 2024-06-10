#!/bin/bash

DATA_DIR="datasets3d/acdc"
JSON_LIST="dataset_acdc_140_20_.json"
SLICE_BS=3

ckpt_paths=("Results/unetr/data_acdc/natural/model_best.pt" "Results/swin_unetr/data_acdc/natural/model_best.pt" "Results/segresnet/data_acdc/natural/model_best.pt" "Results/umamba_bot/data_acdc/natural/model_best.pt" "Results/umamba_enc/data_acdc/natural/model_best.pt" "Results/unet/data_acdc/natural/model_best.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")

echo "==========================================================================================="
echo "==========================================================================================="



echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Inference on ACDC*********************************"



for ((j=0; j<6; j++)); do


  model_name=${model_names[$j]}
  ckpt_path=${ckpt_paths[$j]}



  # check if the checkpoint path exists
  if [ ! -f $ckpt_path ]; then
    echo "Checkpoint path $ckpt_path does not exist. Skipping..."
    continue
  fi

  for ((i=0; i<6; i++)); do


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
  for adv_dir in $surrogate_parent_dir/Adv*/acdc/*; do
    # run code only if the folder exists
    if [ -d "$adv_dir" ]; then

      echo "Evaluating $model_name on Adversarial Images in $adv_dir"
      python inference_on_adv_images.py --model_name $model_name  --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir

      echo "-------------------------------------------------------------------------------------------"
    fi

    done
  done
done

