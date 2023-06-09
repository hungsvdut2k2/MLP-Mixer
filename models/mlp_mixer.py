import torch
import torch.nn.functional as F
from torch import nn


class Patches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size = x.shape[0]
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.reshape(batch_size, 1, patches.shape[-1], patches.shape[1])
        return patches


class MixerBlock(nn.Module):
    def __init__(self, s: int, c: int, ds: int, dc: int, activation=nn.GELU()):
        super().__init__()
        self.activation_layer = activation
        self.weight1 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(s, ds)), requires_grad=True
        )
        self.weight2 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(ds, s)), requires_grad=True
        )
        self.weight3 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(c, dc)), requires_grad=True
        )
        self.weight4 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(dc, c)), requires_grad=True
        )

    def forward(self, x):
        # token-mixing layer
        x_t = torch.permute(x, dims=(0, 1, 3, 2))
        w1_x = x_t @ self.weight1
        w2_x = w1_x @ self.weight2
        w2_x = self.activation_layer(w2_x)
        w2_x = torch.permute(w2_x, dims=(0, 1, 3, 2))
        # skip-connection
        u = w2_x + x
        # channel-mixing layer
        w3_x = u @ self.weight3
        w4_x = w3_x @ self.weight4
        w4_x = self.activation_layer(w4_x)
        # skip-connection
        y = w4_x + u

        return y


class MlpMixer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        s: int,
        c: int,
        ds: int,
        dc: int,
        num_mlp_blocks: int,
        num_classes: int,
    ):
        super().__init__()
        self.c = c
        self.s = s
        self.ds = ds
        self.dc = dc
        self.num_classes = num_classes
        self.layer_norm = nn.LayerNorm([1, s, c])
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(s, c, ds, dc) for i in range(num_mlp_blocks)]
        )
        self.num_classes = num_classes
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2))
        self.patches_extract = Patches(patch_size)

    def forward(self, x):
        patches = self.patches_extract(x)
        for block in self.mixer_blocks:
            patches = self.layer_norm(patches)
            patches = block(patches)
        output = self.classifier(patches)
        if self.num_classes == 2:
            output = nn.Linear(self.c * self.s, 1)(output)
            output = nn.Sigmoid()(output)
        else:
            output = nn.Linear(self.c * self.s, self.num_classes)(output)
            output = nn.Softmax()(output)
        return output
