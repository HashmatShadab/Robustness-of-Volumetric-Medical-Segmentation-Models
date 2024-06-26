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

import sys
import numpy as np
import torch



from monai.utils.misc import fall_back_tuple
from monai.data.utils import dense_patch_slices

def get_slices(input_shape, roi_size):
    # input_shape = (B,C,H,W,D)
    # roi_size = (roi_x, roi_y, roi_z)
    num_spatial_dims = len(input_shape) - 2
    image_size = input_shape[2:]
    roi_size = fall_back_tuple(roi_size, image_size)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    scan_interval = roi_size
    # store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    return slices


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def import_args(source_parser, target_parser):
    for action in source_parser._actions:
        # Avoid adding help argument
        if action.option_strings != ['-h', '--help']:
            target_parser._add_action(action)


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out




class MyOutput():
    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, "a")
    def write(self, text):
            self.stdout.write(text)
            self.log.write(text)
            self.log.flush()
    def close(self):
            self.stdout.close()
            self.log.close()
    def flush(self):
        pass




def print_attack_info(args):
    if args.attack_name=="pgd":           print(f"\n PGD: alpha={args.alpha} , eps={args.eps} , steps={args.steps} , targeted={args.targeted}\n")
    if args.attack_name=="fgsm":          print(f"\n FGSM: eps={args.eps} , targeted={args.targeted}\n")
    if args.attack_name=="bim":           print(f"\n BIM: alpha={args.alpha} , eps={args.eps} , steps={args.steps} , targeted={args.targeted}\n")
    if args.attack_name=="gn":            print(f"\n GN: std={args.std}\n")
    if args.attack_name=="vafa-2d":       print(f"\n VAFA_2D:  q_max={args.q_max} , steps={args.steps} , patch_size={tuple(args.block_size)}\n")
    if args.attack_name=="vafa-3d":       print(f"\n VAFA_3D:  q_max={args.q_max} , steps={args.steps} , patch_size={tuple(args.block_size)} , use_ssim_loss={args.use_ssim_loss}\n")


def get_folder_name(args):
    if   args.attack_name == "pgd"    : folder_name = f"pgd_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
    elif args.attack_name == "fgsm"   : folder_name = f"fgsm_eps_{args.eps}"
    elif args.attack_name == "bim"    : folder_name = f"bim_alpha_{args.alpha}_eps_{args.eps}_i_{args.steps}"
    elif args.attack_name == "gn"     : folder_name = f"gn_std_{args.std}"
    elif args.attack_name == "vafa-2d": folder_name = f"vafa_q_max_{args.q_max}_i_{args.steps}_2d_dct_{args.block_size[0]}x{args.block_size[1]}"
    elif args.attack_name == "vafa-3d": folder_name = f"vafa_q_max_{args.q_max}_i_{args.steps}_3d_dct_{args.block_size[0]}x{args.block_size[1]}x{args.block_size[2]}_use_ssim_loss_{args.use_ssim_loss}"
    else: raise ValueError(f"Attack '{args.attack_name}' is not implemented.")
    return folder_name
