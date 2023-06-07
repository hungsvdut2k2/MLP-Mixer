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
        self.layer_norm = nn.LayerNorm(normalized_shape=(128, 1024))
        self.activation_layer = activation
        self.weight1 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(s, ds))
        )
        self.weight2 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(ds, s))
        )
        self.weight3 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(c, dc))
        )
        self.weight4 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(dc, c))
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
        self.num_classes = num_classes
        super().__init__()
        self.mixer_blocks = [MixerBlock(s, c, ds, dc) for i in range(num_mlp_blocks)]
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(c * s, num_classes), nn.Softmax()
        )
        self.patches_extract = Patches(patch_size)

    def forward(self, x):
        patches = self.patches_extract(x)
        for block in self.mixer_blocks:
            patches = block(patches)
        output = self.classifier(patches)
        return output
