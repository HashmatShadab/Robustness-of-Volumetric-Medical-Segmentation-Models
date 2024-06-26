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



import os
import shutil
import time
import json

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather


from attacks import vafa
from attacks.pgd import projected_gradient_descent_l_inf as pgd_l_inf
from attacks.fgsm import fast_gradient_sign_method_l_inf as fgsm_l_inf
from attacks.bim import basic_iterative_method_l_inf as bim_l_inf
from attacks.gn import gaussain_noise as gn
from attacks.utils import get_target_labels


from attacks.vafa.compression import block_splitting_3d, block_splitting_2d 
import torch_dct as dct_pack

from monai.data import decollate_batch
from helpers import MetricLogger


import cProfile, pstats, io
from pstats import SortKey
def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper





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



def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, logger=None):

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Training Epoch: [{epoch}/{args.max_epochs}]"
    

    for idx, batch_data in enumerate(metric_logger.log_every(loader, 1, header, logger=logger)):
        
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")

        data, target = data.to(device), target.to(device)

        # adversarial training
        if args.adv_training_mode:

            # put model into evaluation mode
            model.eval()

            # set model gradients to None
            for param in model.parameters(): param.grad = None

            images = data
            labels = get_target_labels() if args.targeted else target
            loss_fn  = loss_func


            if args.attack_name=="pgd":
                at_images = pgd_l_inf(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps/255, device=device, targeted=args.targeted, verbose=False)
            elif args.attack_name=="fgsm":
                at_images = fgsm_l_inf(model, images, labels, loss_fn, eps=args.eps/255, device=device, targeted=args.targeted, verbose=False)
            elif args.attack_name=="bim":
                at_images = bim_l_inf(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps/255, device=device, targeted=args.targeted, verbose=False)
            elif args.attack_name=="gn":
                at_images = gn(images, std=args.std/255, device=device, verbose=False)
            elif args.attack_name=="vafa-2d":
                VAFA_2D_Attack = vafa.VAFA_2D(model, loss_fn, batch_size=images.shape[0], q_max=args.q_max, block_size=args.block_size, verbose=False)
                at_images, at_labels, q_tables = VAFA_2D_Attack(images, labels)
            elif args.attack_name=="vafa-3d":
                VAFA_3D_Attack = vafa.VAFA(model, loss_fn, batch_size=images.shape[0], q_max=args.q_max, block_size=args.block_size,
                                           vafa_norm=args.vafa_norm, use_ssim_loss=args.use_ssim_loss, verbose=False)
                at_images, at_labels, q_tables = VAFA_3D_Attack(images, labels)  
            else:
                raise ValueError(f"Attack '{args.attack_name}' is not implemented.")

            data = at_images

            # put model into training mode
            model.train()
        

        # set model gradients to None
        for param in model.parameters(): param.grad = None

        
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        #  Synchronize
        torch.cuda.synchronize()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            metric_logger.update(loss=np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size*args.world_size)

        else:
            metric_logger.update(loss=loss.item(), n=args.batch_size)

        # Add LRs to the metric logger ass well
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    for param in model.parameters(): param.grad = None
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_epoch_freq_reg(model, loader, optimizer, scaler, epoch, loss_func, args, logger=None):


    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Training (Freq. Reg) Epoch: [{epoch}/{args.max_epochs}]"
    


    for idx, batch_data in enumerate(metric_logger.log_every(loader, 1, header, logger=logger)):
    
        if isinstance(batch_data, list):
            data_clean, target_clean = batch_data
        else:
            data_clean, target_clean = batch_data["image"], batch_data["label"]

        device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
        data_clean, target_clean = data_clean.to(device), target_clean.to(device)


        # put model into evaluation mode
        model.eval()

        # set model gradients to None
        for param in model.parameters(): param.grad = None


        images = data_clean
        labels = get_target_labels() if args.targeted else target_clean
        loss_fn  = loss_func


        ## generate adversarial version of clean data
        if args.attack_name=="pgd":
            at_images = pgd_l_inf(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps/255, device=device, targeted=args.targeted, verbose=False)
        elif args.attack_name=="fgsm":
            at_images = fgsm_l_inf(model, images, labels, loss_fn, eps=args.eps/255, device=device, targeted=args.targeted, verbose=False)
        elif args.attack_name=="bim":
            at_images = bim_l_inf(model, images, labels, loss_fn, steps=args.steps, alpha=args.alpha, eps=args.eps/255, device=device, targeted=args.targeted, verbose=False)
        elif args.attack_name=="gn":
            at_images = gn(images, std=args.std/255, device=device, verbose=False)
        elif args.attack_name=="vafa-2d":
            VAFA_2D_Attack = vafa.VAFA_2D(model, loss_fn, batch_size=images.shape[0], q_max=args.q_max, block_size=args.block_size, verbose=False)
            at_images, at_labels, q_tables = VAFA_2D_Attack(images, labels)
        elif args.attack_name=="vafa-3d":
            VAFA_3D_Attack = vafa.VAFA(model, loss_fn, batch_size=images.shape[0], q_max=args.q_max, block_size=args.block_size,
                                       vafa_norm=args.vafa_norm, use_ssim_loss=args.use_ssim_loss, verbose=True)
            at_images, at_labels, q_tables = VAFA_3D_Attack(images, labels)  
        else:
            raise ValueError(f"Attack '{args.attack_name}' is not implemented.")

        data_adv = at_images

        # put model into training mode
        model.train()


        for param in model.parameters():  param.grad = None

        with autocast(enabled=args.amp):

            logits_clean = model(data_clean)
            logits_adv = model(data_adv)

            loss_clean = loss_func(logits_clean, target_clean)
            loss_adv = loss_func(logits_adv, target_clean)
            
            # logits_clean_blocks = block_splitting_3d(logits_clean, tuple(args.block_size))   # [B, C, N_Blocks, Block_H, Block_W, Block_D]
            # logits_adv_blocks   = block_splitting_3d(logits_adv, tuple(args.block_size) )    # [B, C, N_Blocks, Block_H, Block_W, Block_D]


            logits_clean_blocks = block_splitting_3d(logits_clean, (96,96,96))   # [B, C, N_Blocks, Block_H, Block_W, Block_D]
            logits_adv_blocks   = block_splitting_3d(logits_adv, (96,96,96) )    # [B, C, N_Blocks, Block_H, Block_W, Block_D]



            dct_logits_clean = dct_pack.dct_3d(logits_clean_blocks, 'ortho')                 # 3D DCT is applied on last three dimensions
            dct_logits_adv   = dct_pack.dct_3d(logits_adv_blocks, 'ortho')                   # 3D DCT is applied on last three dimensions

            # l1_loss = torch.sum(torch.abs(dct_logits_clean-dct_logits_adv))
            l1_loss = torch.sum(torch.abs(dct_logits_clean-dct_logits_adv))/torch.abs(dct_logits_clean).sum()  # normalized l1 distance

            loss = loss_clean + loss_adv + l1_loss


        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Synchronize
        torch.cuda.synchronize()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            metric_logger.update(loss=np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size)
        else:
            metric_logger.update(loss=loss.item(), n=args.batch_size)
            metric_logger.update(l1_loss=l1_loss.item(), n=args.batch_size)
            metric_logger.update(loss_clean=loss_clean.item(), n=args.batch_size)
            metric_logger.update(loss_adv=loss_adv.item(), n=args.batch_size)

        # Add LRs to the metric logger ass well
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        for param in model.parameters(): param.grad = None

        metric_logger.synchronize_between_processes()

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None, logger=None):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            if args.rank == 0:
                logger.info(
                    f"Val {epoch}/{args.max_epochs} {idx}/{len(loader)} Accuracy: {avg_acc} Time: {(time.time() - start_time):.2f}s"
                )

            start_time = time.time()

    return avg_acc


def save_checkpoint(model, epoch, args, filename="model_latest.pt", best_acc=0, epoch_acc=0, optimizer=None, scheduler=None, logger=None):

    model_state_dict = model.state_dict() if not args.distributed else model.module.state_dict()

    save_dict = {"epoch": epoch, "epoch_acc": epoch_acc, "best_acc": best_acc, "model_state_dict": model_state_dict}

    if optimizer is not None:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        save_dict["scheduler_state_dict"] = scheduler.state_dict()

    filename = os.path.join(args.save_model_dir, filename)

    torch.save(save_dict, filename)

    logger.info(f"\nSaving Checkpoint : {filename}")


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    best_acc=0,
    post_label=None,
    post_pred=None, logger=None):

    writer = None


    scaler = None
    if args.amp: scaler = GradScaler()

    val_acc_max = best_acc

    for epoch in range(0, args.max_epochs):

        if epoch < start_epoch:
            scheduler.step(epoch)
            continue
        
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        logger.info(f"{args.rank} {time.ctime()} Epoch: {epoch}")

        epoch_time = time.time()
        

        if args.adv_training_mode and args.freq_reg_mode:
            train_stats  = train_epoch_freq_reg(model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, logger=logger)
        else:
            train_stats  = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, logger=logger)

        log_stats_train = {
            'Epoch': epoch,
            **{f'train_{key}': value for key, value in train_stats.items()},
        }

        if args.rank == 0:
            with open(os.path.join(args.save_model_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")

        b_new_best = False

        if (epoch + 1) % args.val_every == 0:

            if args.distributed:
                torch.distributed.barrier()

            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred, logger=logger)

            if args.rank == 0:
                logger.info(
                    f"Final Validation  {epoch}/{args.max_epochs - 1} Accuracy: {val_avg_acc} Time: {time.time() - epoch_time:.2f}s")
                log_stats_val = {'Epoch': str(epoch), 'Val Accuracy': str(val_avg_acc)}
                with open(os.path.join(args.save_model_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats_val) + "\n")

                if val_avg_acc > val_acc_max:
                    logger.info(f"\nNew Best ({val_acc_max:.6f} --> {val_avg_acc:.6f}). \n")
                    val_acc_max = val_avg_acc
                    b_new_best = True

                    if args.rank == 0 and args.save_model_dir is not None and args.save_checkpoint and not args.debugging:
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max, epoch_acc=val_avg_acc, filename="model_best.pt",
                                        optimizer=optimizer, scheduler=scheduler, logger=logger)
                        logger.info("\n")

            if args.rank == 0 and args.save_model_dir is not None and args.save_checkpoint and not args.debugging:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, epoch_acc=val_avg_acc, filename="model_latest.pt", optimizer=optimizer,
                                scheduler=scheduler, logger=logger)
                logger.info("\n")

                if b_new_best:
                    logger.info("Copying the 'model_latest.pt' to 'model_best.pt' as new best model!!!!\n\n")
                    shutil.copyfile(os.path.join(args.save_model_dir, "model_latest.pt"), os.path.join(args.save_model_dir, "model_best.pt"))


        if scheduler is not None: scheduler.step(epoch)

    logger.info(f"\n\nTraining Finished !, Best Accuracy: {val_acc_max}")

    return val_acc_max











