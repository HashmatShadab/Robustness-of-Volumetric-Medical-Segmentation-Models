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



  echo "Attack on Hecktor with CosPGD eps 4 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name cospgd --eps 4  --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"

  echo "Attack on Hecktor with CosPGD eps 8 $model_name"
  python wb_attack.py  --model_name $model_name  --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name cospgd --eps 8  --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \



done


# use echo to have a clear separation between different experiments

echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Inference on Hecktor*********************************"



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
  for adv_dir in $surrogate_parent_dir/Adv*/hecktor/*cospgd*; do
    # run code only if the folder exists
    if [ -d "$adv_dir" ]; then

      echo "Evaluating $model_name on Adversarial Images in $adv_dir"
      python inference_on_adv_images.py --model_name $model_name --in_channels 1 --out_channel 3 --feature_size=16 --checkpoint_path $ckpt_path \
        --dataset hecktor --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
        --adv_imgs_dir $adv_dir

      echo "-------------------------------------------------------------------------------------------"
    fi

    done
  done
done




