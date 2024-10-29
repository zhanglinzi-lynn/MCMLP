import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class SpatialMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        # self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, expansion_factor)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        # self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        self.spatial_mixer = SpatialMixer(
            num_patches, num_features, expansion_factor
        )
        self.channel_mixer = ChannelMixer(
            num_patches, num_features, expansion_factor
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.spatial_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x


# def check_sizes(image_size, patch_size):
#     sqrt_num_patches, remainder = divmod(image_size, patch_size)
#     assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
#     num_patches = sqrt_num_patches ** 2
#     return num_patches


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.num_features = data.m
        self.num_patches = args.window
        self.expansion_factor = args.expansion
        self.num_layers = args.layers

        # per-patch fully-connected is equivalent to strided conv2d

        self.mixers = nn.Sequential(
            *[
                MixerLayer(self.num_patches, self.num_features, self.expansion_factor)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        # patches = self.patcher(x)
        # batch_size, num_features, _, _ = patches.shape
        # patches = patches.permute(0, 2, 3, 1)
        # patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mixers(x)
        embedding=torch.squeeze(embedding[:,0,:])
        # print("shape of embedding:",embedding.shape)
        # embedding.shape == (batch_size, num_patches, num_features)
        # embedding = embedding.mean(dim=1)
        # logits = self.classifier(embedding)
        return embedding

