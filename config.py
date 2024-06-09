import argparse


def get_model_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="unetr", type=str, help="model name")
    parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
    parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
    parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    parser.add_argument("--res_block", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--conv_block", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--use_checkpoint", default=True, type=lambda x: (str(x).lower() == 'true'), help="use gradient checkpointing to save memory (Swin-UNETR)")

    return parser

def get_dataset_parser():
    parser = argparse.ArgumentParser()

    """
       ========================================================================================
       ============================= DATASET PARAMETERS =======================================
       --dataset (str): Specifies the dataset to use. Default is 'btcv'.
       --data_dir (str): Specifies the path to the dataset. Default is 'datasets3d/btcv-synapse'.
       --json_list (str): Specifies the name of the json file containing the dataset information.
       --use_normal_dataset (bool): If specified, monai Dataset class will be used.

       ============================================================================================

       """
    parser.add_argument('--dataset', type=str, default=r'btcv')
    parser.add_argument('--data_dir', type=str, default=r'datasets3d/btcv-synapse')
    parser.add_argument('--json_list', type=str, default=r'dataset_synapse_18_12.json')

    # Data augmentation parameters
    parser.add_argument("--use_normal_dataset",  default=True, type=lambda x: (str(x).lower() == 'true'), help="use monai Dataset class")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")


    return parser

def get_wandb_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, default="AdvTransferMed3D")
    parser.add_argument('--entity', type=str, default="hashmatshadab")
    parser.add_argument('--wandb_mode', type=str, default="disabled")
    parser.add_argument('--wandb_name', type=str, default="test")

    return parser

def get_distributed_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    return parser

def get_attack_parser():
    """
       ============================================================================================
       =================================== ATTACK PARAMETERS ======================================

       --attack_name (str): Specifies the name of the adversarial attack to use. Default is 'pgd'.

       --steps (int): Defines the number of iterations to generate adversarial example. Default is 20.

       --alpha (float): Step size for the update during the attack. Default value is 0.01.

       --eps (float): Perturbation budget on the scale of [0,255]. Default is 4.

       --std (float): Standard deviation for Gaussian noise on the scale of [0,255]. Default is 4.

       --targeted (bool): If specified, targeted attack will be chosen.

       --q_max (float): Upper bound on quantization table values in VAFA attack. Default is 20.

       --use_ssim_loss (bool): If specified, SSIM loss will be used in adversarial loss.

       --block_size (int): Defines the DCT block size. Default is [8, 8, 8].

       ============================================================================================
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_name", default="vafa-3d", type=str, help="name of adversarial attack")
    parser.add_argument("--vafa_norm", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--steps", default=20, type=int, help="number of iterations to generate adversarial example")
    parser.add_argument("--alpha", default=0.01, type=float, help="step size for update during attack")
    parser.add_argument("--eps", default=4, type=float, help="perturbation budget on the scale of [0,255]")
    parser.add_argument("--std", default=4, type=float, help="standard deviation for gaussian noise on the scale of [0,255]")
    parser.add_argument("--targeted", action='store_true', help="if targeted attack is to be chosen")
    parser.add_argument("--q_max", default=20, type=float, help="upper bound on quantization table values in VAFA attack")
    parser.add_argument("--use_ssim_loss", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--block_size", default=[32, 32, 32], type=int, nargs="+", help="DCT block size")

    return parser

