# code adapted from: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/


import torch
import warnings
import torch.nn.functional as F
import logging
from torch.autograd import Variable

logger = logging.getLogger(__name__)


def projected_gradient_descent_l_inf(model, images, labels, loss_fn, steps=20, alpha=2 / 255, eps=8 / 255, random_start=False, device=None,
                                     targeted=False, verbose=True):
    if verbose:
        logger.info(f"\nPGD: alpha={alpha} , eps={eps * 255} , steps={steps} , targeted={targeted}\n")
        if images.max() > 1 or images.min() < 0: warnings.warn(
            f"PGD Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    adv_images = images.clone().detach()

    if random_start:
        # starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for i in range(steps):

        adv_images.requires_grad = True

        adv_logits = model(adv_images)

        # calculate loss
        if targeted:
            loss = -1 * loss_fn(adv_logits, labels)
        else:
            loss = loss_fn(adv_logits, labels)

        if verbose:
            logger.info(f"Step: {str(i + 1).zfill(3)} ,   Loss: {round(loss.item(), 5):.5f}")

        # update adversarial images

        ###########################################################################
        loss.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()

        # adv_images.grad.zero_()
        ################################################################################

        # grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        #
        # adv_images = adv_images.detach() + alpha*grad.sign()

        delta = torch.clamp(adv_images - images, min=-eps, max=eps)

        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def projected_gradient_descent_l_inf_plus(model, images, labels, loss_fn, steps=20, alpha=2/255, eps=8/255, random_start=True, device=None,
                                     targeted=False, verbose=True, noise_data=None):
    
    if verbose:
        logger.info(f"\nPGD: alpha={alpha} , eps={eps*255} , steps={steps} , targeted={targeted}\n")
        if images.max()>1 or images.min()<0 : warnings.warn(f"PGD Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)


    for i in range(steps):

        noise_batch = Variable(noise_data[0:images.size(0)], requires_grad=True).cuda()
        in1 = images + noise_batch
        in1.clamp_(0, 1)


        adv_logits = model(in1)

        # calculate loss
        if targeted:
            loss = -1*loss_fn(adv_logits, labels)
        else:
            loss = loss_fn(adv_logits, labels)

        if verbose:
            logger.info(f"Step: {str(i+1).zfill(3)} ,   Loss: {round(loss.item(), 5):.5f}")

        # update adversarial images

        ###########################################################################
        loss.backward()

        perturbation = alpha * noise_batch.grad.sign()
        noise_data[0:images.size(0)] += perturbation.data
        noise_data.clamp_(-eps, eps)

        noise_batch.grad.data.zero_()

    adv_images = images + noise_data[0:images.size(0)]
    adv_images.clamp_(0, 1)


    return adv_images, noise_data.detach()


def projected_gradient_descent_l_inf_intermediate(model, images, labels, steps=20, alpha=2 / 255, eps=8 / 255, random_start=True, device=None,
                                     targeted=False, verbose=True, feature_loss="1", momentum=False):
    if verbose:
        logger.info(f"\nPGD: alpha={alpha} , eps={eps * 255} , steps={steps} , targeted={targeted}\n")
        if images.max() > 1 or images.min() < 0: warnings.warn(
            f"PGD Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    if random_start:
        adv = torch.randn(images.shape).to(device)
    else:
        adv = images +  images.new(images.size()).uniform_(-eps, eps)



    """
    SwinUnetr Intermediate features:
    1. encoder0 : B, 48, 96, 96, 96
    2. encoder1 : B, 48, 48, 48, 48
    3. encoder2 : B, 96, 24, 24, 24
    4. encoder3 : B, 192, 12, 12, 12
    5. decoder4:  B, 768, 3, 3, 3
    6. decoder3:  B, 384, 6, 6, 6
    7. decoder2:  B, 192, 12, 12, 12
    8. decoder1:  B, 96, 24, 24, 24
    9. decoder0:  B, 48, 48, 48, 48
    
    loss:
    1. cosine loss on features of encoder3 + decoder4 + decoder3 + decoder2
    2. mse loss on features of encoder3 + decoder4 + decoder3 + decoder2

    """

    adv.requires_grad = True
    adv_noise = 0

    for i in range(steps):

        adv1 = adv + 0

        adv_features, adv_logits = model(adv1, get_feat=True)
        org_features, org_logits = model(images, get_feat=True)

        if feature_loss == "swinunetr_1_cos": #  cosine loss on features of encoder3 + decoder4 + decoder3 + decoder2
            loss = 0
            # flatten the features and normalize, so that cosine similarity can be calculated
            loss += F.cosine_similarity(adv_features[3].view(adv_features[3].shape[0], -1), org_features[3].view(org_features[3].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[4].view(adv_features[4].shape[0], -1), org_features[4].view(org_features[4].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[5].view(adv_features[5].shape[0], -1), org_features[5].view(org_features[5].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[6].view(adv_features[6].shape[0], -1), org_features[6].view(org_features[6].shape[0], -1), dim=1).mean()
            loss = -loss

        elif feature_loss == "swinunetr_1_mse": # mse loss on features of encoder3 + decoder4 + decoder3 + decoder2
            loss = 0
            loss += F.mse_loss(adv_features[4], org_features[4])
            loss += F.mse_loss(adv_features[5], org_features[5])
            loss += F.mse_loss(adv_features[6], org_features[6])
            loss += F.mse_loss(adv_features[7], org_features[7])

        elif feature_loss == "unetr_enc_mse" or feature_loss == "segresnet_enc_mse":
            # first 4 layers
            loss = 0
            loss += F.mse_loss(adv_features[0], org_features[0])
            loss += F.mse_loss(adv_features[1], org_features[1])
            loss += F.mse_loss(adv_features[2], org_features[2])
            loss += F.mse_loss(adv_features[3], org_features[3])

        elif feature_loss == "unetr_enc_cos"  or feature_loss == "segresnet_enc_cos":
            # first 4 layers
            loss = 0
            loss += F.cosine_similarity(adv_features[0].view(adv_features[0].shape[0], -1), org_features[0].view(org_features[0].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[1].view(adv_features[1].shape[0], -1), org_features[1].view(org_features[1].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[2].view(adv_features[2].shape[0], -1), org_features[2].view(org_features[2].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[3].view(adv_features[3].shape[0], -1), org_features[3].view(org_features[3].shape[0], -1), dim=1).mean()

        elif feature_loss == "unetr_dec_mse":
            # last 4 layers
            loss = 0
            loss += F.mse_loss(adv_features[4], org_features[4])
            loss += F.mse_loss(adv_features[5], org_features[5])
            loss += F.mse_loss(adv_features[6], org_features[6])
            loss += F.mse_loss(adv_features[7], org_features[7])

        elif feature_loss == "unetr_dec_cos":
            # last 4 layers
            loss = 0
            loss += F.cosine_similarity(adv_features[4].view(adv_features[4].shape[0], -1), org_features[4].view(org_features[4].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[5].view(adv_features[5].shape[0], -1), org_features[5].view(org_features[5].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[6].view(adv_features[6].shape[0], -1), org_features[6].view(org_features[6].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[7].view(adv_features[7].shape[0], -1), org_features[7].view(org_features[7].shape[0], -1), dim=1).mean()

        elif feature_loss == "segresnet_dec_mse":
            # last 4 layers
            loss = 0
            loss += F.mse_loss(adv_features[4], org_features[4])
            loss += F.mse_loss(adv_features[5], org_features[5])
            loss += F.mse_loss(adv_features[6], org_features[6])

        elif feature_loss == "segresnet_dec_cos":
            # last 4 layers
            loss = 0
            loss += F.cosine_similarity(adv_features[4].view(adv_features[4].shape[0], -1), org_features[4].view(org_features[4].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[5].view(adv_features[5].shape[0], -1), org_features[5].view(org_features[5].shape[0], -1), dim=1).mean()
            loss += F.cosine_similarity(adv_features[6].view(adv_features[6].shape[0], -1), org_features[6].view(org_features[6].shape[0], -1), dim=1).mean()


        else:
            raise ValueError("Invalid feature loss type")

        loss.backward()

        if verbose:
            logger.info(f"Step: {str(i + 1).zfill(3)} ,   Loss: {round(loss.item(), 5):.5f}")

        if momentum:
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3, 4), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        adv.data = adv.data + alpha * adv_noise.sign()
        adv.data = torch.where(adv.data > images.data + eps, images.data + eps, adv.data)
        adv.data = torch.where(adv.data < images.data - eps, images.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()

    logger.info(f"Distance between the images{(adv - images).max() * 255}")
    return adv.data







