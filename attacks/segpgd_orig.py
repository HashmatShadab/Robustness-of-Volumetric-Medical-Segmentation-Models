# Author: Asif Hanif
# Link: https://github.com/asif-hanif/segpgd


# PGD code adapted from: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/
# SegPGD: https://arxiv.org/pdf/2207.12391.pdf (Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr)


import torch
import warnings


def to_onehot(labels, num_classes):
    B, dim, H, W, D = labels.shape
    assert dim == 1, f"Invalid 'labels' shape. The second dimension should have a size of 1, but it has a size of {dim}."
    labels = torch.nn.functional.one_hot(labels[:, 0], num_classes=num_classes)  # [B,1,H,W,D] --> [B,H,W,D] --> [B,H,W,D,num_classes]
    return labels.permute(0, 4, 1, 2, 3)  # [B,H,W,D,num_classes] --> [B,num_classes,H,W,D]


def DiceLoss(input, target, squared_pred=False, smooth_nr=1e-5, smooth_dr=1e-5):
    intersection = torch.sum(target * input)

    if squared_pred:
        ground_o = torch.sum(target ** 2)
        pred_o = torch.sum(input ** 2)
    else:
        ground_o = torch.sum(target)
        pred_o = torch.sum(input)

    denominator = ground_o + pred_o

    dice_loss = 1.0 - (2.0 * intersection + smooth_nr) / (denominator + smooth_dr)

    return dice_loss


def seg_projected_gradient_descent_l_inf(model, images, labels, loss_fn, num_classes=None, steps=20, alpha=2 / 255, eps=8 / 255,
                                         random_start=True, device=None, targeted=False, verbose=True):
    # model for volumetric image segmentation
    # images: [B,C,H,W,D] normalized to [0,1]. B=BatchSize, C=Number-of-Channels,  H=Height,  W=Width, D=Depth
    # labels: [B,1,H,W,D] (in integer form)

    # Note: I used single channel volumetric images. Therefore, C=1

    if verbose:
        print(f"\nSegPGD: alpha={alpha} , eps={eps * 255} , steps={steps} , targeted={targeted}\n")
        if images.max() > 1 or images.min() < 0: warnings.warn(
            f"SegPGD Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    assert num_classes is not None, "'num_classes' is None. Specify the number of classes present in ground truth labels for 'SegPGD' to proceed."

    images = images.clone().detach().to(device)  # [B,C=1,H,W,D]
    labels = labels.clone().detach().to(device)  # [B,1,H,W,D]

    adv_images = images.clone().detach()

    if random_start:
        # starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    softmax = torch.nn.Softmax(dim=1)

    for i in range(steps):

        adv_images.requires_grad = True

        adv_logits = model(adv_images)  # adv_images[B,C=1,H,W,D] --> adv_logits[B,NumClass,H,W,D]

        pred_labels = torch.argmax(adv_logits, dim=1)  # [B,NumClass,H,W,D] --> [B,H,W,D]

        correct_voxels = labels[:, 0] == pred_labels  # correctly classified voxels  [B,H,W,D]
        wrong_voxels = labels[:, 0] != pred_labels  # wrongly classified voxels    [B,H,W,D]

        # calculate number of correct and wrong voxels
        num_correct_voxels = torch.sum(correct_voxels).item()
        num_wrong_voxels = torch.sum(wrong_voxels).item()

        labels_onehot = to_onehot(labels.long(), num_classes=num_classes)  # [B,1,H,W,D] -->  [B,NumClass,H,W,D]

        adv_pred_softmax = softmax(adv_logits)  # [B,NumClass,H,W,D]

        adv_pred_softmax = adv_pred_softmax.permute(1, 0, 2, 3, 4)  # [B,NumClass,H,W,D] -->  [NumClass,B,H,W,D]
        labels_onehot = labels_onehot.permute(1, 0, 2, 3, 4)  # [B,NumClass,H,W,D] -->  [NumClass,B,H,W,D]

        # calculate loss
        loss_correct = DiceLoss(adv_pred_softmax[:, correct_voxels], labels_onehot[:, correct_voxels], squared_pred=True)

        loss_wrong = DiceLoss(adv_pred_softmax[:, wrong_voxels], labels_onehot[:, wrong_voxels], squared_pred=True)

        lmbda = 1 #i / (2 * steps)

        # loss = (1 - lmbda) * loss_correct + lmbda * loss_wrong
        loss =  lmbda * loss_wrong
        print("Correct Voxels:", num_correct_voxels, " , Wrong Voxels:", num_wrong_voxels, " , lmbda:", lmbda, " , Loss:", loss.item())

        if targeted:
            loss = -1 * loss

        # if verbose:
        #     if i == 0 or (i + 1) % 10 == 0: print("Step:", str(i + 1).zfill(3), " ,   Loss:", f"{round(loss.item(), 5):3.5f}")

        # update adversarial images
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()

        delta = torch.clamp(adv_images - images, min=-eps, max=eps)

        adv_images = torch.clamp(images + delta, min=0, max=1).detach()


    return adv_images

