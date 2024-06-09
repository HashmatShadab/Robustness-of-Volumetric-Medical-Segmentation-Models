#!/bin/bash

DATA_DIR="datasets3d/acdc"
JSON_LIST="dataset_acdc_140_20_.json"
SLICE_BS=3
# Baseline Training of models on ACDC and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on ACDC

ckpt_paths=("Results/unetr/data_acdc/natural/model_best.pt" "Results/swin_unetr/data_acdc/natural/model_best.pt" "Results/segresnet/data_acdc/natural/model_best.pt" "Results/umamba_bot/data_acdc/natural/model_best.pt" "Results/umamba_enc/data_acdc/natural/model_best.pt" "Results/unet/data_acdc/natural/model_best.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")

echo "==========================================================================================="
echo "==========================================================================================="

echo "********************************Running Attack on ACDC*********************************"


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


  echo "Attack on ACDC with PGD eps 4 $model_name"
  python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name pgd --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "Attack on ACDC with PGD eps 8 $model_name"
  python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name pgd --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"



  echo "Attack on ACDC with FGSM eps 4 $model_name"
  python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name fgsm --eps 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"


  echo "Attack on ACDC with FGSM eps 8 $model_name"
  python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name fgsm --eps 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

  echo "-------------------------------------------------------------------------------------------"



  echo "Attack on ACDC with VAFA-3D with qmax 10 $model_name"
  python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 4  --std 4 --q_max 10 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

  echo "Attack on ACDC with VAFA-3D with qmax 20 $model_name"
  python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 8 --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

  echo "Attack on ACDC with VAFA-3D with qmax 30 $model_name"
  python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
  --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
  --attack_name vafa-3d --eps 8 --std 8 --q_max 30 --steps 20 --block_size 32 32 32 --use_ssim_loss True \

  echo "-------------------------------------------------------------------------------------------"

  done


# use echo to have a clear separation between different experiments

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

    echo "Attack on ACDC with GN std 4 $model_name"
    python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST  --slice_batch_size $SLICE_BS \
    --attack_name gn --eps 4  --std 4 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

    echo "-------------------------------------------------------------------------------------------"

    #
    echo "Attack on ACDC with GN std 8 $model_name"
    python wb_attack.py  --model_name $model_name   --in_channels 1 --out_channel 4 --feature_size=16 --checkpoint_path $ckpt_path \
    --dataset acdc --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size $SLICE_BS  \
    --attack_name gn --eps 8 --std 8 --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss False \

    echo "-------------------------------------------------------------------------------------------"

  done


