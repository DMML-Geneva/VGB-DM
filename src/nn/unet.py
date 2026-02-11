import torch
from torch import nn
from torch.nn import functional as F
import torchvision


class Block2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder2d(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block2d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            # print(x.shape)
            x = block(x)
            # print(x.shape)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder2d(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), cond_size=0):
        super().__init__()
        self.chs = chs
        cond_sizes = [0] * len(chs)
        cond_sizes[0] = cond_size
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    chs[i] + cond_sizes[i], chs[i + 1], 2, stride=2
                )
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [Block2d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(
            enc_ftrs
        )  # .squeeze(3)
        return enc_ftrs


class UNet2d(nn.Module):
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
        num_class=1,
        retain_dim=True,
    ):
        super().__init__()
        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.final_act = nn.Sigmoid()
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)
