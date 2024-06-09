#!/bin/bash

DATA_DIR="datasets3d/1_1_1_s176v2"
JSON_LIST="dataset_hecktor.json"
SLICE_BS=3
# Baseline Training of models on Hecktor and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on Hecktor

#ckpt_paths=("Results_hecktor_normal_training/unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/swin_unetr/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/segresnet/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_bot/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_enc/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/unet/data_hecktor/natural/model_latest.pt")
#model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")


ckpt_paths=("Results_hecktor_normal_training/umamba_bot/data_hecktor/natural/model_latest.pt" "Results_hecktor_normal_training/umamba_enc/data_hecktor/natural/model_latest.pt")
model_names=("umamba_bot" "umamba_enc")



echo "==========================================================================================="
echo "==========================================================================================="

echo "********************************Running Attack on Hecktor*********************************"



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

  python viz.py --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
  --attack_name vafa-3d --q_max 30 --steps 20 --block_size 32 32 32 --use_ssim_loss True

  python viz.py --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
  --attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss True

  python viz.py --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
  --attack_name vafa-3d --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss True


  parent_dir=$(dirname $ckpt_path)

  # Loop over all the folders in parent directory starting with "qtables"

  for qtable_dir in $parent_dir/qtables*/hecktor/vafa*; do
    # run code only if the folder exists
    if [ -d "$qtable_dir" ]; then

      echo "Evaluating $model_name on Adversarial Images in $qtable_dir"
      python plot_qtable.py --dataset_path $qtable_dir --filter_size 2
      python plot_qtable.py --dataset_path $qtable_dir --filter_size 4
      python plot_qtable.py --dataset_path $qtable_dir --filter_size 8
    fi
  done




done



