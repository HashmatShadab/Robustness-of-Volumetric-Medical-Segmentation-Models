# code adapted from: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/
# CosPGD: https://arxiv.org/pdf/2302.02213.pdf

import torch
import warnings

from monai.networks import one_hot

def cosine_projected_gradient_descent_l_inf(model, images, labels, loss_fn, n_classes=None, steps=20, alpha=2/255, eps=8/255, random_start=True, device=None, targeted=False, verbose=True):
    
    if verbose:
        print(f"\nCosPGD: alpha={alpha} , eps={eps*255} , steps={steps} , targeted={targeted}\n")
        if images.max()>1 or images.min()<0 : warnings.warn(f"CosPGD Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    adv_images = images.clone().detach()

    if random_start:
        # starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps,eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    softmax = torch.nn.Softmax(dim=1)
    
    for i in range(steps):

        adv_images.requires_grad = True

        adv_logits = model(adv_images)


        labels_onehot = one_hot(labels, num_classes=n_classes) # [B,1,H,W,D] -->  [B,NumClass,H,W,D]

        pred_softmax  = softmax(adv_logits)                    # [B,NumClass,H,W,D]

        cos_sim = cosine_similarity(pred_softmax, labels_onehot).mean()
        

        # calculate loss
        if targeted:
            loss = -1*loss_fn(adv_logits, labels)*(1-cos_sim)
            alpha = -1*alpha
        else:
            loss = loss_fn(adv_logits, labels)*cos_sim


        if verbose: 
            if i==0 or (i+1)%10 == 0: print("Step:", str(i+1).zfill(3), " ,   Loss:", f"{round(loss.item(),5):3.5f}" )


        # update adversarial images
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()

        delta = torch.clamp(adv_images - images, min=-eps, max=eps)

        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images







