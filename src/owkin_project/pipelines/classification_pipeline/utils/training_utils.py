import torch
from torch import nn, optim
import numpy as np

def get_lr_scheduler(optimizer, warmup, max_iters):
    #return CosineWarmupScheduler(optimizer, warmup, max_iters)
    p = nn.Parameter(torch.empty(4, 4))
    opt = optim.Adam([p], lr=1e-3)
    return optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.65 ** epoch)

def weighted_loss(weight: float):
    bce_loss = nn.BCELoss(reduction='none')
    def loss(y_pred, y_true):
        intermediate_loss = bce_loss(y_pred, y_true)
        return torch.mean(weight * y_true * intermediate_loss + (1 - y_true) * intermediate_loss)
    return loss

def warmup_lr(optimizer, start_lr: float, end_lr: float, epoch_max: int)        :
    step_lr = (end_lr - start_lr)/epoch_max
    lr = optimizer.param_groups[0]['lr']
    lr += step_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #return optimizer

def set_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #return optimizer

def get_optimizer(model, lr: float = 0.000001):
    return optim.Adam(model.parameters(), lr=lr)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor