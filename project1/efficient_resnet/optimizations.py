import numpy as np

import torch
import torchvision.transforms as transforms

# -------------------------------
# Mixup Functions
# -------------------------------
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------------
# Data Transforms: AutoAugment, Mixup, Cutout, etc.
# -------------------------------
def get_transforms(config):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]
    transform_list = []
    if config['training'].get('autoaugment', False):
        transform_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
    if config['training'].get('data_augmentation', False):
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])
    transform_list.append(transforms.ToTensor())
    if config['training'].get('normalization', True):
        transform_list.append(transforms.Normalize(mean, std))
    if config['training'].get('cutout', False):
        transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
    return transforms.Compose(transform_list)