from abc import ABC

from torch import nn
import torch

__all__ = ["RegLoss"]


class RegLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction='sum')

    def forward(self, regs, gt_regs, mask):
        mask = mask[:, :, None].expand_as(gt_regs).float()
        loss = sum(self.loss(r * mask, gt_regs * mask) / (mask.sum() + 1e-4) for r in regs)
        return loss / len(regs)
