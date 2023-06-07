import torch
import torch.nn.functional as F
from torch import nn


class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return batch_size


class MixerLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


class MlpMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return
