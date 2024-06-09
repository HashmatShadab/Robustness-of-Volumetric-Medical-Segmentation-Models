# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import argparse
import os
import sys
import json
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import wandb

from load_surrogate_models import get_unetr_model, get_swin_unetr_model, get_segresnet_model, get_unet_model
# from mamba_models import get_emunet_3d, get_lmaunet_3d, get_nnmamba_3d, get_segmamba_3d, get_umamba_bot_3d, get_umamba_enc_3d

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training



# from datasets.acdc import get_loader_acdc
from datasets.btcv import get_loader_btcv, get_loader_acdc
from datasets.hecktor import get_loader_hecktor
from datasets.abdomen import get_loader_abdomen

from utils.utils import MyOutput
from utils.utils import print_attack_info
from utils.utils import get_folder_name

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from utils.utils import get_slices, import_args

from config import get_dataset_parser, get_wandb_parser, get_distributed_parser, get_attack_parser, get_model_args

def get_args() -> argparse.Namespace:
    """
    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Training on Surrogate Models')


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

    parser.add_argument("--use_pretrained", default=False, type=lambda x: (str(x).lower() == 'true'), help="model will be initialized from saved pre-trained checkpoint.")
    parser.add_argument("--pretrained_path", default="", type=str, help="full path of pre-trained checkpoint")
    parser.add_argument("--resume", default=True, type=lambda x: (str(x).lower() == 'true'), help="resume training from a checkpoint")
    parser.add_argument("--resume_latest", default=True, type=lambda x: (str(x).lower() == 'true'), help="resume training from latest checkpoint")
    parser.add_argument("--resume_best",default=False, type=lambda x: (str(x).lower() == 'true'), help="resume training from best checkpoint")
    parser.add_argument("--resume_but_restart", default=False, type=lambda x: (str(x).lower() == 'true'), help="resume training from the checkpoint but set start_epoch=0")

    parser.add_argument("--logdir", default="None", type=str, help="directory to save the tensorboard logs")

    parser.add_argument("--save_checkpoint", default=True, type=lambda x: (str(x).lower() == 'true'), help="save checkpoint during training")

    parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
    parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size (during inference)")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", default=False, type=lambda x: (str(x).lower() == 'true'), help="do not use amp for training")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")

    # different modes
    parser.add_argument("--gen_train_adv_mode", default=False, type=lambda x: (str(x).lower() == 'true'), help="if adversarial versions of train samples are to be generated")
    parser.add_argument("--gen_val_adv_mode", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="if adversarial versions of validation/test samples are to be generated")
    parser.add_argument("--adv_training_mode",default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="if adversarial training is to be performed. adv-images are created during training.")
    parser.add_argument("--freq_reg_mode", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="adversarial training with frequency regularization term in loss function...")

    # directories
    parser.add_argument("--adv_images_dir", default="", type=str, help="parent directory containing adversarial images")
    parser.add_argument("--save_adv_images_dir", default=None, type=str, help="parent directory to save adversarial images")
    parser.add_argument("--save_model_dir", default="Results", type=str,
                        help="parent directory to save model finetuned on adversarial images")
    parser.add_argument("--no_sub_dir_model", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="if mentioned, sub-folder will not be searched in parent direcotry containing model checkpoint")
    parser.add_argument("--no_sub_dir_adv_images", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="if mentioned, sub-folder will not be searched in parent direcotry containing adv-images")

    parser.add_argument("--debugging", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="if mentioned, folders would not be created and results will not be saved.")

    args = parser.parse_args()

    args.block_size = tuple(args.block_size)

    # sanity checks on arguments
    assert not (args.resume_latest and not args.resume), "To resume from last checkpoint, '--resume' has to be also True"
    assert not (args.resume_best and not args.resume), "To resume from best checkpoint, '--resume' has to be also True"
    assert not (
                args.resume_latest and args.resume_best), "'--resume_latest' and '--resume_best' are mutually exclusive. Use either of them."
    assert not (
                args.freq_reg_mode and not args.adv_training_mode), "To use frequency-regularization in adversarial training, '--adv_training_mode' must be mentioned"

    return args

def get_log_name(args):
    if args.adv_training_mode:
        if   args.attack_name == "pgd"    : folder_name = f"pgd_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
        elif args.attack_name == "fgsm"   : folder_name = f"fgsm_eps_{args.eps}"
        elif args.attack_name == "bim"    : folder_name = f"bim_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
        elif args.attack_name == "gn"     : folder_name = f"gn_std_{args.std}"
        elif args.attack_name == "vafa-2d": folder_name = f"vafa2d_q_max_{args.q_max}_i_{args.steps}_2d_dct_{args.block_size[0]}x{args.block_size[1]}"
        elif args.attack_name == "vafa-3d": folder_name = f"vafa3d_q_max_{args.q_max}_i_{args.steps}_3d_dct_{args.block_size[0]}x{args.block_size[1]}x{args.block_size[2]}_use_ssim_loss_{args.use_ssim_loss}"
        else: raise ValueError(f"Attack '{args.attack_name}' is not implemented.")
    else:
        folder_name = "natural"
    return folder_name


def main():
    now_start = datetime.now() 

    args = get_args()

    args.amp = not args.noamp
    args.now_start = now_start

    args.save_model_dir = os.path.join(args.save_model_dir, f"{args.model_name}",
                                     f"data_{args.dataset}", get_log_name(args))
    args.folder_name = args.save_model_dir
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir, exist_ok=True)
        # save argparse file content
        with open(f"{os.path.join(args.save_model_dir, 'args.json')}", 'wt') as f:
            json.dump(vars(args),f, indent=4, default=str)

    log_name = os.path.join(args.save_model_dir, "train.log")
    logging.basicConfig(filename=log_name, filemode="a", format="%(name)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logger.info(f"Eval logs stored in {log_name}")
    logger.info(args)

    wandb.init(project=args.project, entity=args.entity, mode=args.wandb_mode,
               name=args.wandb_name)


    # log will not be saved if debugging
    if not args.debugging:
        # keep the terminal output on console and also save it to a file
        sys.stdout = MyOutput(f"{os.path.join(args.save_model_dir, 'log.out' )}")


    print("\n\n", "".join(["#"]*130), "\n", "".join(["#"]*130), "\n\n""")


    if args.adv_training_mode:
        logger.info(f"Adversarial-Training of '{args.model_name.upper()}' Model under following Attack:")
        print_attack_info(args)
    else:
        logger.info(f"\nTraining the '{args.model_name.upper()}'  Model ... ")



    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("\nNum. of GPUs = ", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args, logger=logger)


def main_worker(gpu, args, logger):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

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

    logger.info(f"\nRank = {args.rank} ,  GPU = {args.gpu}")

    if args.rank == 0: logger.info(f"BatchSize: {args.batch_size}, Epochs: {args.max_epochs}\n")

    inf_size = [args.roi_x, args.roi_y, args.roi_z]


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
                                dropout_rate=args.dropout_rate)
    elif args.model_name == "swin_unetr":

        model = get_swin_unetr_model(num_classes=args.out_channels, in_channels=args.in_channels,
                                     img_size=(args.roi_x, args.roi_y, args.roi_z),
                                     feature_size=48,
                                     drop_rate=args.dropout_rate,

                                     )
    elif args.model_name == "unet":
        model = get_unet_model(num_classes=args.out_channels, in_channels=args.in_channels, dropout_prob=args.dropout_rate)
    elif args.model_name == "segresnet":
        model = get_segresnet_model(num_classes=args.out_channels, in_channels=args.in_channels, dropout_prob=args.dropout_rate)
    elif args.model_name == "emunet":
        model = get_emunet_3d(input_channels=args.in_channels, out_channels=args.out_channels, dropout_rate=args.dropout_rate)
    elif args.model_name == "lmaunet":
        model = get_lmaunet_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    elif args.model_name == "nnmamba":
        model = get_nnmamba_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    elif args.model_name == "segmamba":
        model = get_segmamba_3d(input_channels=args.in_channels, num_classes=args.out_channels, dropout_rate=args.dropout_rate)
    elif args.model_name == "umamba_bot":
        model = get_umamba_bot_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    elif args.model_name == "umamba_enc":
        model = get_umamba_enc_3d(input_channels=args.in_channels, num_classes=args.out_channels)
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    # Compute number of parameters of the model
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Model Parameters = {model_total_params:,}\n")
    

    start_epoch = 0
    best_acc = 0


    if args.use_pretrained:
        pretrained_path  = args.pretrained_path
        checkpoint_dict = torch.load(pretrained_path)
        model.load_state_dict(checkpoint_dict["model_state_dict"] if "model_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["state_dict"])
        logger.info(f"\nLoading Pre-trained Model Weights from:  {pretrained_path}\n")


    if args.resume and os.path.isfile(os.path.join(args.save_model_dir, 'model_latest.pt')):


        if args.resume_latest: checkpoint_path  = os.path.join(args.save_model_dir, 'model_latest.pt')
        if args.resume_best: checkpoint_path  = os.path.join(args.save_model_dir, 'model_best.pt')

        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

        model_weights = checkpoint_dict["model_state_dict"] if "model_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["state_dict"]
        msg = model.load_state_dict(model_weights)

        start_epoch = checkpoint_dict["epoch"]+1
        best_acc = checkpoint_dict["best_acc"]

        logger.info(f"\nResuming Training ...")
        logger.info(f"Resume Checkpoint Path: {checkpoint_path}")
        logger.info(f"Model Loaded with message: {msg}")
        logger.info(f"Start Epoch={start_epoch}")
        if "epoch_acc" in checkpoint_dict.keys(): logger.info(f"Accuracy (at Epoch={start_epoch-1})={checkpoint_dict['epoch_acc']:0.6f}")
        logger.info(f"Best Accuracy={best_acc:0.6f}\n")
        

        pretrained_path = checkpoint_path
    else:
        logger.info(f"\nTraining from Scratch ...")
    

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap)



    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Model Parameters = {model_total_params:,}\n")

    model.cuda(args.gpu)


    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)


    ## optimizer (scale learning rate by teh batch size, if batch size is 3, then don't scale)

    logger.info(f"Original Learning Rate = {args.optim_lr}")
    scale_lr = args.batch_size / 3
    args.optim_lr = args.optim_lr * scale_lr
    logger.info(f"Scaling Learning Rate  to {args.optim_lr} using scale factor = {scale_lr}")

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight)
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))


    # load optimizer state if resume
    if args.resume and os.path.isfile(os.path.join(args.save_model_dir, 'model_latest.pt')):
        logger.info(f"Loading optimizer state_dict from: {pretrained_path}")
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"] if "optimizer_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["optimizer"])


    ## scheduler
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None


    # load scheduler state if resume
    if args.resume and os.path.isfile(os.path.join(args.save_model_dir, 'model_latest.pt')):
        logger.info(f"Loading scheduler state_dict from: {pretrained_path}")
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"] if "scheduler_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["scheduler"])


    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)

    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred  = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc   = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)


    accuracy = run_training(
        model=model,
        train_loader=loaders[0],
        val_loader=loaders[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_acc = best_acc,
        post_label=post_label,
        post_pred=post_pred, logger=logger)

    logger.info(f"\n{'#' * 130}\n{'#' * 130}")

    if args.adv_training_mode:
        logger.info(f"\n Adversarial-Training of '{args.model_name.upper()}' Model completed under following Attack:")
        print_attack_info(args)

        logger.info(f" Adversarially Trained Model Weights Saved at Path: {args.folder_name}")


    now_end = datetime.now()
    logger.info(f'\nTime & Date  =  {now_end.strftime("%I:%M %p, %d_%b_%Y")}\n')

    duration = now_end - args.now_start
    duration_in_s = duration.total_seconds() 

    days    = divmod(duration_in_s, 86400)       # Get days (without [0]!)
    hours   = divmod(days[1], 3600)              # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)               # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)              # Use remainder of minutes to calc seconds

    logger.info(f"Total Time =>   {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds \n\n")

    logger.info(f"\n{'#' * 130}\n{'#' * 130}")

    return accuracy

    


if __name__ == "__main__":
    main()
