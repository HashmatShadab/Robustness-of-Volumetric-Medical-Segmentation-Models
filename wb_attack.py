import argparse
import logging
import os
from datetime import datetime

import lpips
import monai
import nibabel as nib
import numpy as np
import torch
import wandb
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
import json

from attacks import vafa
from attacks.bim import basic_iterative_method_l_inf as bim_l_inf
from attacks.fgsm import fast_gradient_sign_method_l_inf as fgsm_l_inf
from attacks.gn import gaussain_noise as gn
from attacks.pgd import projected_gradient_descent_l_inf as pgd_l_inf
from attacks.pgd import projected_gradient_descent_l_inf_intermediate as pgd_l_inf_intermediate
from attacks.pgd import projected_gradient_descent_l_inf_plus as pgd_l_inf_plus
from attacks.cospgd import cosine_projected_gradient_descent_l_inf as cospgd_l_inf
from attacks.segpgd_orig import seg_projected_gradient_descent_l_inf
# from datasets.acdc import get_loader_acdc
from datasets.btcv import get_loader_btcv, get_loader_acdc
from datasets.hecktor import get_loader_hecktor
from datasets.abdomen import get_loader_abdomen
from load_surrogate_models import get_unetr_model, get_swin_unetr_model, get_segresnet_model, get_unet_model
# from mamba_models import get_emunet_3d, get_lmaunet_3d, get_nnmamba_3d, get_segmamba_3d, get_umamba_bot_3d, get_umamba_enc_3d
from utils.utils import get_slices, import_args

from config import get_dataset_parser, get_wandb_parser, get_distributed_parser, get_attack_parser, get_model_args

def get_args() -> argparse.Namespace:
    """
    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Attack on Surrogate Models')


    parser1 = get_dataset_parser()
    import_args(parser1, parser)

    parser2 = get_wandb_parser()
    import_args(parser2, parser)

    parser3 = get_distributed_parser()
    import_args(parser3, parser)

    parser4 = get_model_args()
    import_args(parser4, parser)

    parser5 = get_attack_parser()
    import_args(parser5, parser)

    # model name and checkpoint path
    parser.add_argument('--checkpoint_path', type=str, default='surrogate_weig')

    # save adversarial images
    parser.add_argument('--save_adv_imgs_dir', type=str, default=r'Adv_images')
    parser.add_argument('--debugging', action='store_true', help="if debugging mode is to be chosen")

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

    # pgd intermediate attack parameters
    parser.add_argument("--feature_loss", default="1", type=str, help="loss type for pgd intermediate attack")
    parser.add_argument("--momentum", default=True, type=lambda x: (str(x).lower() == 'true'), help="momentum for pgd intermediate attack")
    parser.add_argument("--random_start", default=False, type=lambda x: (str(x).lower() == 'true'), help="for pgd intermediate attack")

    parser.add_argument("--slice_batch_size", default=6, type=int, help="number of slices taken at a time for attack")





    args = parser.parse_args()

    return args

def get_log_name(args):
    if   args.attack_name == "pgd"    : folder_name = f"pgd_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
    elif   args.attack_name == "segpgd"    : folder_name = f"segpgd_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"

    elif args.attack_name == "pgd_plus": folder_name = f"pgd_plus_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
    elif args.attack_name == "pgd_intermediate": folder_name = f"pgd_intermediate_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}_random_start_{args.random_start}_momentum_{args.momentum}_feature_loss_{args.feature_loss}"
    elif args.attack_name == "fgsm"   : folder_name = f"fgsm_eps_{args.eps}"
    elif args.attack_name == "bim"    : folder_name = f"bim_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
    elif args.attack_name == "gn"     : folder_name = f"gn_std_{args.std}"
    elif args.attack_name == "vafa-2d": folder_name = f"vafa2d_q_max_{args.q_max}_i_{args.steps}_2d_dct_{args.block_size[0]}x{args.block_size[1]}"
    elif args.attack_name == "vafa-3d": folder_name = f"vafa3d_q_max_{args.q_max}_i_{args.steps}_3d_dct_{args.block_size[0]}x{args.block_size[1]}x{args.block_size[2]}_use_ssim_loss_{args.use_ssim_loss}"
    elif args.attack_name == "cospgd" : folder_name = f"cospgd_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
    else: raise ValueError(f"Attack '{args.attack_name}' is not implemented.")
    return folder_name


if __name__ == "__main__":

    now_start = datetime.now()

    args = get_args()
    # from args.checkpoint_path get the path to parent directory
    parent_dir = os.path.dirname(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint_epoch = checkpoint["epoch"]
    del checkpoint

    save_adv_imgs_dir = os.path.join(parent_dir, f"{args.save_adv_imgs_dir}_Epoch_{checkpoint_epoch}", f"{args.dataset}", get_log_name(args))
    if not os.path.exists(save_adv_imgs_dir): os.makedirs(save_adv_imgs_dir, exist_ok=True)

    log_name = os.path.join(save_adv_imgs_dir, f"attack.log")

    logging.basicConfig(filename=log_name, filemode="a", format="%(name)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logger.info(f"Eval logs stored in {log_name}")
    logger.info(f"Adversarial images stored at {save_adv_imgs_dir}")
    logger.info(args)

    wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode,
               name=args.wandb_name)

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

    logger.info(f"Device: {device}")


    if load_weights_with_chk_path:
        # check if checkpoint file exists, if not then load model from scratch
        if not os.path.isfile(args.checkpoint_path):
            logger.info(f"Model: {args.model_name} loading weights are not loaded from {args.checkpoint_path} as file does not exist.")
        else:
            logger.info(f"Model: {args.model_name} loading weights from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            checkpoint_epoch = checkpoint["epoch"]
            msg = model.load_state_dict(checkpoint["model_state_dict"], strict=False) if "model_state_dict" in checkpoint.keys() else model.load_state_dict(
                checkpoint["state_dict"], strict=False)
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

    for i, batch in enumerate(loaders):
        # if i >0: break

        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

        img_name = os.path.basename(batch["image"].meta["filename_or_obj"][0])
        lbl_name = os.path.basename(batch["label"].meta["filename_or_obj"][0])

        logger.info(f"\n\n\nAdversarial Attack on Image: {img_name} \n")

        input_shape = val_inputs.shape
        roi_size = (96, 96, 96)
        slices = get_slices(input_shape, roi_size)

        logger.info(f'Created {len(slices)} slices of size {roi_size} from input volume of size {input_shape}.')

        slice_batch_size = 1  if args.attack_name == "pgd_plus" else args.slice_batch_size # number of slices in one batch
        logger.info(f"Slice Batch Size {slice_batch_size}")

        adv_val_inputs = torch.zeros(input_shape).to(device)
        slice_noise_data = torch.zeros(slice_batch_size, 1, 96, 96, 96).to(device)

        for start in range(0, len(slices), slice_batch_size):
            stop = min(start + slice_batch_size, len(slices))

            logger.info(f"\nSlice No. = {start + 1}-to-{stop} of {len(slices)}")

            slice_data = [val_inputs[0, 0][slices[j]].unsqueeze(0).unsqueeze(1) for j in range(start, stop)]  # [B, 1, 96, 96, 96]
            slice_data = torch.cat(slice_data, 0) if len(slice_data) > 1 else slice_data[0]

            # actual labels of the slice
            slice_labels = [val_labels[0, 0][slices[j]].unsqueeze(0).unsqueeze(1) for j in range(start, stop)]  # [B, 1, 96, 96, 96]
            slice_labels = torch.cat(slice_labels, 0) if len(slice_labels) > 1 else slice_labels[0]

            images = slice_data
            labels = slice_labels

            ## generate adversarial version of clean data
            if args.attack_name == "pgd_plus":
                at_images, noise  = pgd_l_inf_plus(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps / 255.0,
                                      device=device, targeted=args.targeted, verbose=True, noise_data=slice_noise_data.clone())
                slice_noise_data = noise

            elif args.attack_name == "pgd":
                at_images = pgd_l_inf(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps / 255.0,
                                      device=device, targeted=args.targeted, verbose=True)
            elif args.attack_name == "cospgd":
                at_images = cospgd_l_inf(model, images, labels, loss_fn, n_classes=args.out_channels,  steps=args.steps, alpha=args.alpha, eps=args.eps / 255.0,
                                      device=device, targeted=args.targeted, verbose=True)
            elif args.attack_name == "segpgd":
                at_images = seg_projected_gradient_descent_l_inf(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps / 255.0,
                                      device=device, targeted=args.targeted, verbose=True, num_classes=args.out_channels)
            elif args.attack_name == "pgd_intermediate":
                at_images = pgd_l_inf_intermediate(model, images, labels, steps=args.steps, alpha=args.alpha, eps=args.eps / 255.0,
                                      device=device, targeted=args.targeted, verbose=True, feature_loss=args.feature_loss, momentum=args.momentum,
                                                   random_start=args.random_start)
            elif args.attack_name == "fgsm":
                at_images = fgsm_l_inf(model, images, labels, loss_fn, eps=args.eps / 255.0, device=device, targeted=args.targeted,
                                       verbose=True)
            elif args.attack_name == "bim":
                at_images = bim_l_inf(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps / 255.0,
                                      device=device, targeted=args.targeted, verbose=True)
            elif args.attack_name == "gn":
                at_images = gn(images, std=args.std / 255.0, device=device, verbose=True)
            elif args.attack_name == "vafa-2d":
                VAFA_2D_Attack = vafa.VAFA_2D(model, loss_fn, batch_size=images.shape[0], q_max=args.q_max, block_size=args.block_size,
                                              verbose=True, steps=args.steps)
                at_images, at_labels, q_tables = VAFA_2D_Attack(images, labels)
            elif args.attack_name == "vafa-3d":
                VAFA_3D_Attack = vafa.VAFA(model, loss_fn, batch_size=images.shape[0], q_max=args.q_max, block_size=args.block_size,
                                           use_ssim_loss=args.use_ssim_loss, steps=args.steps, verbose=True)
                at_images, at_labels, q_tables = VAFA_3D_Attack(images, labels)
            else:
                raise ValueError(f"Attack '{args.attack_name}' is not implemented.")

            # adv_val_inputs[0,0][slices[j]] = at_images
            for counter, j in enumerate(range(start, stop)): adv_val_inputs[0, 0][slices[j]] = at_images[counter].unsqueeze(0)

        # inference on whole volume of input data
        with torch.no_grad():
            # inference on clean inputs
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

            logger.info(f"\nImageName={img_name}")
            logger.info(f"Adv Attack Success Rate (voxel): {round(voxel_suc_rate * 100, 3)}  (%)")
            logger.info(
                f"Mean Organ Dice (Clean): {round(np.nanmean(dice_organ_dict_clean[img_name]) * 100, 2):.2f} (%)        Mean Organ HD95 (Clean): {round(np.nanmean(hd95_organ_dict_clean[img_name]), 2)}")
            logger.info(
                f"Mean Organ Dice (Adv)  : {round(np.nanmean(dice_organ_dict_adv[img_name]) * 100, 2):.2f} (%)        Mean Organ HD95 (Adv)  : {round(np.nanmean(hd95_organ_dict_adv[img_name]), 2)}")
            logger.info(f"LPIPS_Alex: {round(lpips_alex_dict[img_name], 4)}")
            logger.info('\n\n')


        ## saving images
        if not args.debugging:

            clean_save_images_dir = os.path.join(save_adv_imgs_dir, 'imagesTrClean' if args.gen_train_adv_mode else 'imagesTsClean')
            clean_save_labels_dir = os.path.join(save_adv_imgs_dir, 'labelsTrClean' if args.gen_train_adv_mode else 'labelsTsClean')
            adv_save_images_dir = os.path.join(save_adv_imgs_dir, 'imagesTrAdv' if args.gen_train_adv_mode else 'imagesTsAdv')

            if not os.path.exists(clean_save_images_dir):  os.makedirs(clean_save_images_dir, exist_ok=True)
            if not os.path.exists(clean_save_labels_dir):  os.makedirs(clean_save_labels_dir, exist_ok=True)
            if not os.path.exists(adv_save_images_dir):    os.makedirs(adv_save_images_dir, exist_ok=True)

            ## save clean images
            img_clean = nib.Nifti1Image((val_inputs[0, 0].cpu().numpy() * 255).astype(np.uint8),
                                        np.eye(4))  # save axis for data (just identity)
            img_clean.header.get_xyzt_units()
            img_clean.to_filename(os.path.join(clean_save_images_dir, 'clean_' + img_name));
            logger.info(f"Image=clean_{img_name} saved at: {clean_save_images_dir}")

            ## save clean ground truth labels
            lables_clean = nib.Nifti1Image((batch["label"][0, 0].cpu().numpy()).astype(np.float32), np.eye(4))
            lables_clean.to_filename(os.path.join(clean_save_labels_dir, lbl_name));
            logger.info(f"Labels={lbl_name} saved at: {clean_save_labels_dir}")

            ## save adversarial images
            img_adv = nib.Nifti1Image((adv_val_inputs[0, 0].cpu().numpy() * 255).astype(np.uint8),
                                      np.eye(4))  # save axis for data (just identity)
            img_adv.header.get_xyzt_units()
            img_adv.to_filename(os.path.join(adv_save_images_dir, 'adv_' + img_name));
            logger.info(f"Image=adv_{img_name} saved at: {adv_save_images_dir}")

    dice_clean_all = []
    dice_adv_all = []
    for key in dice_organ_dict_clean.keys(): dice_clean_all.append(np.nanmean(dice_organ_dict_clean[key]))
    for key in dice_organ_dict_adv.keys(): dice_adv_all.append(np.nanmean(dice_organ_dict_adv[key]))

    hd95_clean_all = []
    hd95_adv_all = []
    for key in hd95_organ_dict_clean.keys(): hd95_clean_all.append(np.nanmean(hd95_organ_dict_clean[key]))
    for key in hd95_organ_dict_adv.keys(): hd95_adv_all.append(np.nanmean(hd95_organ_dict_adv[key]))

    logger.info(f"\n{'#' * 130}\n{'#' * 130}")

    logger.info(" Model Weights Path:", )
    logger.info(f"\n Dataset = {args.dataset.upper()}")

    # if not args.debugging: logger.info(f"\n Path of Adversarial Images = {}")

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
                    "LPIPS_Alex": np.mean(lpips_alex_all),}

    attack_result_file_path = os.path.join(save_adv_imgs_dir, f"dataset_{args.dataset}_surrogate_{args.model_name}_{get_log_name(args)}.txt")
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
