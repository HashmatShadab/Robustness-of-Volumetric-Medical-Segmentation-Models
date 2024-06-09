import argparse
import logging
import os
from datetime import datetime

import lpips
import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

from config import get_dataset_parser, get_wandb_parser, get_distributed_parser, get_model_args
# from datasets.acdc import get_loader_acdc
from datasets.btcv import get_loader_btcv, get_loader_acdc
from datasets.hecktor import get_loader_hecktor
from datasets.abdomen import get_loader_abdomen
from load_surrogate_models import get_unetr_model, get_swin_unetr_model, get_segresnet_model, get_unet_model
# from mamba_models import get_emunet_3d, get_lmaunet_3d, get_nnmamba_3d, get_segmamba_3d, get_umamba_bot_3d, get_umamba_enc_3d

from attacks.vafa.compression import block_splitting_3d
import torch_dct
import json
from utils.utils import import_args


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Transfer towards Black-box Domain')

    parser1 = get_wandb_parser()
    import_args(parser1, parser)

    parser2 = get_dataset_parser()
    import_args(parser2, parser)

    parser3 = get_distributed_parser()
    import_args(parser3, parser)

    parser4 = get_model_args()
    import_args(parser4, parser)

    """
    ================================================================================================================
    =================================== MODE PARAMETERS ============================================
    --gen_train_adv_mode (bool): If True, training data is loaded and adversarial versions of training samples are generated.
    --gen_val_adv_mode (bool): If True, validation data is loaded and adversarial versions of validation samples are generated.
    --test_mode (bool): If True, test validation is loaded and adversarial versions of test samples are generated.
    ================================================================================================================
    """

    parser.add_argument("--gen_train_adv_mode", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="if adversarial versions of train samples are to be generated")
    parser.add_argument("--gen_val_adv_mode", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="if adversarial versions of validation/test samples are to be generated")
    parser.add_argument("--test_mode", default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument("--freq_mode", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--filter", default="low", type=str, help="filter type")
    parser.add_argument("--filter_size", default=8, type=int, help="filter size")

    # model parameters
    parser.add_argument('--checkpoint_path', type=str, default=r'F:\Results_abdomen\unet\model_best.pt')

    parser.add_argument('--adv_imgs_dir', type=str,
                        default=r'F:\Code\Projects\AdvTransferMed3D\Results\unetr\data_abdomen\natural\Adv_images_Epoch_3\abdomen\fgsm_eps_8.0')
    parser.add_argument("--slice_batch_size", default=3, type=int, help="number of slices taken")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    now_start = datetime.now()

    args = get_args()
    adv_imgs_dir = args.adv_imgs_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint_epoch = checkpoint["epoch"]
    del checkpoint

    parent_dir = os.path.dirname(args.checkpoint_path)
    save_adv_imgs_dir = os.path.join(parent_dir, f"{args.dataset}")
    if not os.path.exists(save_adv_imgs_dir): os.makedirs(save_adv_imgs_dir, exist_ok=True)

    log_path = os.path.join(save_adv_imgs_dir, f"xxx_eval_target_model_{args.model_name}_Epoch_{checkpoint_epoch}.log")
    logging.basicConfig(filename=log_path, filemode="a", format="%(name)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if args.freq_mode:
        logger.info(f"Frequency Mode: {args.filter} filter of size {args.filter_size} is applied.")
        attack_result_file_path = os.path.join(save_adv_imgs_dir,
                                               f"eval_model_{args.model_name}_epoch{checkpoint_epoch}_freq_{args.filter}_size_{args.filter_size}.txt")
    else:
        attack_result_file_path = os.path.join(save_adv_imgs_dir,
                                               f"xxx_eval_model_{args.model_name}_epoch{checkpoint_epoch}.txt")



    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # add console handler to logger
    logger.addHandler(ch)

    logger.info(f"Eval logs stored in {log_path}")
    logger.info(f"Attack Result File Path: {attack_result_file_path}")
    logger.info(f"Prediction stored in {save_adv_imgs_dir}")

    logger.info(args)

    if args.dataset == "acdc":
        loaders = get_loader_acdc(args)
        args.out_channels = 4
    elif args.dataset == "btcv":
        loaders = get_loader_btcv(args)
        args.out_channels = 14
    elif args.dataset == "hecktor":
        loaders = get_loader_hecktor(args)
        args.out_channels = 3
    elif args.dataset == "abdomen":
        loaders = get_loader_abdomen(args)
        args.out_channels = 14
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not implemented.")

    logger.info(f"\nDataset = {args.dataset.upper()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_weights_with_chk_path = True

    if args.model_name == "unetr":
        model = get_unetr_model(in_channels=args.in_channels,
                                num_classes=args.out_channels,
                                img_size=(args.roi_x, args.roi_y, args.roi_z),
                                feature_size=args.feature_size,
                                hidden_size=args.hidden_size,
                                mlp_dim=args.mlp_dim,
                                num_heads=args.num_heads,
                                proj_type=args.pos_embed,
                                norm_name=args.norm_name,
                                conv_block=True,
                                res_block=True,
                                dropout_rate=0.0)
    elif args.model_name == "swin_unetr":

        model = get_swin_unetr_model(num_classes=args.out_channels, in_channels=args.in_channels,
                                     img_size=(args.roi_x, args.roi_y, args.roi_z),
                                     feature_size=48,
                                     drop_rate=0.0,

                                     )
    elif args.model_name == "unet":
        model = get_unet_model(num_classes=args.out_channels, in_channels=args.in_channels, dropout_prob=0.0)
    elif args.model_name == "segresnet":
        model = get_segresnet_model(num_classes=args.out_channels, in_channels=args.in_channels, dropout_prob=0.0)

    elif args.model_name == "emunet":
        model = get_emunet_3d(input_channels=args.in_channels, out_channels=args.out_channels, dropout_rate=0.0)
    elif args.model_name == "lmaunet":
        model = get_lmaunet_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    elif args.model_name == "nnmamba":
        model = get_nnmamba_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    elif args.model_name == "segmamba":
        model = get_segmamba_3d(input_channels=args.in_channels, num_classes=args.out_channels, dropout_rate=0.0)
    elif args.model_name == "umamba_bot":
        model = get_umamba_bot_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    elif args.model_name == "umamba_enc":
        model = get_umamba_enc_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    if load_weights_with_chk_path:
        # check if checkpoint file exists, if not then load model from scratch
        if not os.path.isfile(args.checkpoint_path):
            logger.info(f"Model: {args.model_name} loading weights are not loaded from {args.checkpoint_path} as file does not exist.")
        else:
            logger.info(f"Model: {args.model_name} loading weights from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            msg = model.load_state_dict(checkpoint["model_state_dict"],
                                        strict="False") if "model_state_dict" in checkpoint.keys() else model.load_state_dict(
                checkpoint["state_dict"], strict="False")
            logger.info(msg)

    model.eval()
    model.to(device)

    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    loss_fn = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)

    transform_true_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    transform_pred_label = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)

    dice_score_monai = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    hd95_score_monai = HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False,
                                               reduction=MetricReduction.MEAN, get_not_nans=True)

    dice_organ_dict_clean = {}
    dice_organ_dict_adv = {}

    hd95_organ_dict_clean = {}
    hd95_organ_dict_adv = {}

    lpips_alex_dict = {}

    voxel_success_rate_list = []
    logger.info("\n\n")

    slice_batch_size = args.slice_batch_size
    for i, batch in enumerate(loaders):
        # if i >0: break

        # get clean images
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())  # Image [Min,Max]=[0,1]

        img_name = os.path.basename(batch["image"].meta["filename_or_obj"][0])
        lbl_name = os.path.basename(batch["label"].meta["filename_or_obj"][0])

        ## load adv-image
        adv_val_inputs_path = os.path.join(adv_imgs_dir, "imagesTsAdv", f"adv_{img_name}")
        logger.info(f"Image Name = {img_name}\nLoading Adversarial Image :{adv_val_inputs_path}")
        adv_val_inputs = nib.load(adv_val_inputs_path).get_fdata() / 255.0  # Image Shape=[H,W,D]   [Min,Max]=[0,1]
        adv_val_inputs = torch.tensor(adv_val_inputs).unsqueeze(0).unsqueeze(0).to(device,
                                                                                   dtype=torch.float32)  # Image Shape=[B,C,H,W,D]   [Min,Max]=[0,1]

        # TO DO: think about overlaping regions

        # inference on whole volume of input data
        with torch.no_grad():
            # inference on clean inputs
            if args.freq_mode:
                dct_val_inputs = torch_dct.dct_3d(val_inputs, norm='ortho')
                dct_adv_val_inputs = torch_dct.dct_3d(adv_val_inputs, norm='ortho')
                # create a mask to filter out high frequency components using a low pass filter
                mask = torch.zeros_like(dct_val_inputs)
                if args.filter == "low":
                    mask[:, :, :args.filter_size, :args.filter_size, :args.filter_size] = 1
                else:
                    mask[:, :, args.filter_size:, args.filter_size:, args.filter_size:] = 1
                dct_val_inputs = dct_val_inputs * mask
                dct_adv_val_inputs = dct_adv_val_inputs * mask
                val_inputs = torch_dct.idct_3d(dct_val_inputs, norm='ortho')
                adv_val_inputs = torch_dct.idct_3d(dct_adv_val_inputs, norm='ortho')

            val_logits = sliding_window_inference(val_inputs, (96, 96, 96), slice_batch_size, model, overlap=args.infer_overlap)
            val_scores = torch.softmax(val_logits, 1).cpu().numpy()
            val_labels_clean = np.argmax(val_scores, axis=1).astype(np.uint8)

            # inference on adversarial inputs
            val_logits_adv = sliding_window_inference(adv_val_inputs, (96, 96, 96), slice_batch_size, model, overlap=args.infer_overlap)
            val_scores_adv = torch.softmax(val_logits_adv, 1).cpu().numpy()
            val_labels_adv = np.argmax(val_scores_adv, axis=1).astype(np.uint8)

            # ture labels
            val_labels = val_labels.cpu().numpy().astype(np.uint8)[0]

            ## Ground Truth
            val_true_labels_list = decollate_batch(batch["label"].cuda())
            val_true_labels_convert = [transform_true_label(val_label_tensor) for val_label_tensor in val_true_labels_list]

            ## Clean Predictions
            val_clean_pred_labels_list = decollate_batch(val_logits)
            val_clean_pred_labels_convert = [transform_pred_label(val_pred_tensor) for val_pred_tensor in val_clean_pred_labels_list]

            ## Adv Predictions
            val_adv_pred_labels_list = decollate_batch(val_logits_adv)
            val_adv_pred_labels_convert = [transform_pred_label(val_pred_tensor) for val_pred_tensor in val_adv_pred_labels_list]

            ## MONAI DICE Score
            dice_clean = dice_score_monai(y_pred=val_clean_pred_labels_convert, y=val_true_labels_convert)
            dice_adv = dice_score_monai(y_pred=val_adv_pred_labels_convert, y=val_true_labels_convert)

            dice_organ_dict_clean[img_name] = dice_clean[0].tolist()
            dice_organ_dict_adv[img_name] = dice_adv[0].tolist()

            ## MONAI HD95 Score
            hd95_score_clean = hd95_score_monai(y_pred=val_clean_pred_labels_convert, y=val_true_labels_convert)
            hd95_score_adv = hd95_score_monai(y_pred=val_adv_pred_labels_convert, y=val_true_labels_convert)

            hd95_organ_dict_clean[img_name] = hd95_score_clean[0].tolist()
            hd95_organ_dict_adv[img_name] = hd95_score_adv[0].tolist()

            img = val_inputs[0, 0].permute(2, 0, 1).unsqueeze(1).float().cpu()
            adv = adv_val_inputs[0, 0].permute(2, 0, 1).unsqueeze(1).float().cpu()
            lpips_alex_dict[img_name] = 1 - loss_fn_alex((2 * img - 1), (2 * adv - 1)).view(-1, ).mean().item()

            voxel_suc_rate = (val_labels_clean != val_labels_adv).sum() / np.prod(val_labels_clean.shape)
            voxel_success_rate_list.append(voxel_suc_rate)

            logger.info(f"Adv Attack Success Rate (voxel): {round(voxel_suc_rate * 100, 3)}  (%)")
            logger.info(
                f"Mean Organ Dice (Clean): {round(np.nanmean(dice_organ_dict_clean[img_name]) * 100, 2):.2f} (%)        Mean Organ HD95 (Clean): {round(np.nanmean(hd95_organ_dict_clean[img_name]), 2)}")
            logger.info(
                f"Mean Organ Dice (Adv)  : {round(np.nanmean(dice_organ_dict_adv[img_name]) * 100, 2):.2f} (%)        Mean Organ HD95 (Adv)  : {round(np.nanmean(hd95_organ_dict_adv[img_name]), 2)}")
            logger.info(f"LPIPS_Alex: {round(lpips_alex_dict[img_name], 4)}")
            logger.info("\n\n")

        ## saving images

        adv_save_images_dir = os.path.join(save_adv_imgs_dir, 'predsTsAdv')
        clean_save_images_dir = os.path.join(save_adv_imgs_dir, 'predsTsclean')

        if not os.path.exists(adv_save_images_dir):    os.makedirs(adv_save_images_dir, exist_ok=True)
        if not os.path.exists(clean_save_images_dir):    os.makedirs(clean_save_images_dir, exist_ok=True)

        ## save clean predictions
        clean_pred = torch.argmax(val_clean_pred_labels_convert[0], dim=0)
        clean_pred = nib.Nifti1Image((clean_pred.cpu().numpy()).astype(np.float32), np.eye(4))
        clean_pred.to_filename(os.path.join(clean_save_images_dir, "advpred_" + lbl_name));
        logger.info(f"Labels=advpred_{lbl_name} saved at: {clean_save_images_dir}")

        ## save adversarial predictions
        adv_pred = torch.argmax(val_adv_pred_labels_convert[0], dim=0)
        adv_pred = nib.Nifti1Image((adv_pred.cpu().numpy()).astype(np.float32), np.eye(4))
        adv_pred.to_filename(os.path.join(adv_save_images_dir, "advpred_" + lbl_name));
        logger.info(f"Labels=advpred_{lbl_name} saved at: {adv_save_images_dir}")

    dice_clean_all = []
    dice_adv_all = []
    for key in dice_organ_dict_clean.keys(): dice_clean_all.append(np.nanmean(dice_organ_dict_clean[key]))
    for key in dice_organ_dict_adv.keys(): dice_adv_all.append(np.nanmean(dice_organ_dict_adv[key]))

    hd95_clean_all = []
    hd95_adv_all = []
    for key in hd95_organ_dict_clean.keys(): hd95_clean_all.append(np.nanmean(hd95_organ_dict_clean[key]))
    for key in hd95_organ_dict_adv.keys(): hd95_adv_all.append(np.nanmean(hd95_organ_dict_adv[key]))

    logger.info(f"\n Model = {args.model_name.upper()} \n")
    logger.info(" Model Weights Path:", )
    logger.info(f"\n Dataset = {args.dataset.upper()}")

    logger.info(f"\n Path of Adversarial Images = {adv_imgs_dir}")

    logger.info("\n Attack Info:")

    logger.info('\n')
    logger.info(f" Overall Mean Dice (Clean): {round(np.mean(dice_clean_all) * 100, 3):0.3f}  (%)")
    logger.info(f" Overall Mean Dice (Adv)  : {round(np.mean(dice_adv_all) * 100, 3):0.3f}  (%)")

    logger.info('\n')
    logger.info(f" Overall Mean HD95 (Clean): {round(np.mean(hd95_clean_all), 3):0.3f}")
    logger.info(f" Overall Mean HD95 (Adv)  : {round(np.mean(hd95_adv_all), 3):0.3f}")

    lpips_alex_all = []
    for key in lpips_alex_dict.keys(): lpips_alex_all.append(lpips_alex_dict[key])

    logger.info('\n')
    logger.info(f" Overall LPIPS_Alex: {round(np.mean(lpips_alex_all), 4):0.4f}")

    now_end = datetime.now()
    logger.info(f'\n Time & Date  =  {now_end.strftime("%I:%M %p")} , {now_end.strftime("%d_%b_%Y")}\n')

    attack_stats = {"Clean Dice": np.mean(dice_clean_all), "Adv Dice": np.mean(dice_adv_all),
                    "Clean HD95": np.mean(hd95_clean_all), "Adv HD95": np.mean(hd95_adv_all),
                    "LPIPS_Alex": np.mean(lpips_alex_all), }

    with open(attack_result_file_path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(attack_stats) + "\n")

    duration = now_end - now_start
    duration_in_s = duration.total_seconds()

    days = divmod(duration_in_s, 86400)  # Get days (without [0]!)
    hours = divmod(days[1], 3600)  # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)  # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)  # Use remainder of minutes to calc seconds

    logger.info(
        f" Total Time =>   {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds \n\n")

    logger.info(f"{'#' * 130}\n{'#' * 130}\n")
    logger.info(" Done!\n")

    logger.info('\n')
    logger.info(
        f" Overall Mean Dice (Clean): {round(np.mean(dice_clean_all) * 100, 3):0.3f}  (%), Overall Mean Dice (Adv)  : {round(np.mean(dice_adv_all) * 100, 3):0.3f}  (%)")
    logger.info(
        f" Overall Mean HD95 (Clean): {round(np.mean(hd95_clean_all), 3):0.3f}, Overall Mean HD95 (Adv)  : {round(np.mean(hd95_adv_all), 3):0.3f}")
