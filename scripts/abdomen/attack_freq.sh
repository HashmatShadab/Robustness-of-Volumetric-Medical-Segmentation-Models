#!/bin/bash

DATA_DIR="datasets3d/abdomenCT_resampled"
JSON_LIST="dataset.json"
SLICE_BS=3
# Baseline Training of models on Abdomen CT and Adv. Training (PGD-K, FGSM, GN, VAFA-3D) on Abdomen CT

ckpt_paths=("Results/unetr/data_abdomen/natural/model_latest.pt" "Results/swin_unetr/data_abdomen/natural/model_latest.pt" "Results/segresnet/data_abdomen/natural/model_latest.pt" "Results/umamba_bot/data_abdomen/natural/model_latest.pt" " Results_no_amp/umamba_enc/data_abdomen/natural/model_latest.pt"  "Results/unet/data_abdomen/natural/model_latest.pt")
model_names=("unetr" "swin_unetr" "segresnet" "umamba_bot" "umamba_enc" "unet")

# use echo to have a clear separation between different experiments

echo "==========================================================================================="
echo "==========================================================================================="
echo "********************************Running Attack on Abdomen CT*********************************"



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
  for adv_dir in $model_parent_dir/Adv*/abdomen/*eps_8*; do
    # run code only if the folder exists
    if [ -d "$adv_dir" ]; then

      echo "Evaluating $model_name on Adversarial Images in $adv_dir"
      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 8

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 16

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 32

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 2 --upper_limit 64

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 8 --upper_limit 96

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 16 --upper_limit 96

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 32 --upper_limit 96


      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 16 --upper_limit 48


      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 32 --upper_limit 64


      echo "-------------------------------------------------------------------------------------------"
    fi

    done


  for adv_dir in $model_parent_dir/Adv*/abdomen/*q_max_30*; do
    # run code only if the folder exists
    if [ -d "$adv_dir" ]; then

      echo "Evaluating $model_name on Adversarial Images in $adv_dir"
      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 8

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 16

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 0 --upper_limit 32

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 2 --upper_limit 64

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 8 --upper_limit 96

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 16 --upper_limit 96

      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 32 --upper_limit 96


      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 16 --upper_limit 48


      python inference_on_adv_images_freq.py --model_name $model_name   --in_channels 1 --out_channel 14 --feature_size=16 --checkpoint_path $ckpt_path \
      --dataset abdomen --data_dir=$DATA_DIR --json_list $JSON_LIST --slice_batch_size 6  \
      --adv_imgs_dir $adv_dir --filter band --lower_limit 32 --upper_limit 64


      echo "-------------------------------------------------------------------------------------------"
    fi

    done


done

