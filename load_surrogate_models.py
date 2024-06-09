from collections import OrderedDict
import numpy as np
import torch
import logging

from monai.networks.nets import UNETR
from monai.networks.nets import SwinUNETR
from monai.networks.nets import SegResNet
from monai.networks.nets import UNet

def get_unetr_model(num_classes, in_channels=1, img_size=(96, 96, 96),
                    feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                    proj_type='conv', norm_name='instance', res_block=True,
                    conv_block=True,
                    dropout_rate=0.0
                    ):

    model = UNETR(
        in_channels=in_channels,
        out_channels=num_classes,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        proj_type=proj_type,
        norm_name=norm_name,
        res_block=res_block,
        conv_block=conv_block,
        dropout_rate=dropout_rate,
        spatial_dims=len(img_size),
    )

    return model

def get_swin_unetr_model(num_classes, in_channels=1, img_size=(96, 96, 96), feature_size=48,
                         drop_rate=0.0, dropout_path_rate=0.0):

    """
    Define SwinUNETR model arguments
     def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:

    """
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=num_classes,
        img_size=img_size,
        feature_size=feature_size,
        drop_rate=drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=True,
        spatial_dims=len(img_size),
    )
    return model

def get_segresnet_model(num_classes, in_channels=1, img_size=(96, 96, 96), dropout_prob=0.0):
    model = SegResNet(
        in_channels=in_channels,
        out_channels=num_classes,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=32,
        dropout_prob=dropout_prob,
        spatial_dims=len(img_size),
    )

    return model

def get_unet_model(in_channels=1,
              num_classes=14, dropout_prob=0.0,
                ):

    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=dropout_prob,

    )

    return model




if __name__ == '__main__':
    # Load all supreme models
    model = get_unet_model(num_classes=14)
    # print total number of parameters
    print(sum(p.numel() for p in model.parameters()))
    print(f"Total number of parameters UNet: {sum(p.numel() for p in model.parameters())}")

    model = get_segresnet_model(num_classes=14)
    # print total number of parameters
    print(sum(p.numel() for p in model.parameters()))
    print(f"Total number of parameters SegResNet: {sum(p.numel() for p in model.parameters())}")

    model = get_swin_unetr_model(num_classes=14)
    # print total number of parameters
    print(sum(p.numel() for p in model.parameters()))
    print(f"Total number of parameters SwinUNETR: {sum(p.numel() for p in model.parameters())}")

    model = get_unetr_model(num_classes=14)
    # print total number of parameters
    print(sum(p.numel() for p in model.parameters()))
    print(f"Total number of parameters UNETR: {sum(p.numel() for p in model.parameters())}")


