"""
This file is a almost a copy-paste form my 2026 PA "Astronomical Adaptive Optics Point Spread Function Reconstruction with Machine Learning Algorithms" (HES-SO master, 1001, Lausanne) with some minor adjustments to make it work with pytorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Dice coefficient metric for evaluating segmentation performance
def dice_coef(y_true, y_pred):
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)

    return (2. * intersection + 1) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 1)

def mse_loss(y_true, y_pred):
    mse = nn.MSELoss()
    return mse(y_pred, y_true)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def double_conv(in_channels, out_channels):
    return DoubleConv(in_channels, out_channels)

def inconv(in_channels, out_channels):
    return double_conv(in_channels, out_channels)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

def down(in_channels, out_channels):
    return Down(in_channels, out_channels)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            conv_in_channels = out_channels + skip_channels

        self.conv = DoubleConv(conv_in_channels, out_channels)

    # Handle cropping/padding to match shape
    def crop_to_match(self, x, skip):
        # Crop or pad `x` to match `skip`
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y >= 0 and diff_x >= 0:
            x = F.pad(x, [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2
            ])
        else:
            crop_top = (-diff_y) // 2
            crop_bottom = (-diff_y) - crop_top
            crop_left = (-diff_x) // 2
            crop_right = (-diff_x) - crop_left

            x = x[
                :,
                :,
                crop_top:x.size(2) - crop_bottom,
                crop_left:x.size(3) - crop_right
            ]

        return x

    def forward(self, x, skip):
        x = self.up(x)
        x = self.crop_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

def up(in_channels, skip_channels, out_channels, bilinear=True):
    return Up(in_channels, skip_channels, out_channels, bilinear)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)

def outconv(in_channels, out_channels):
    return OutConv(in_channels, out_channels)

class UNet(nn.Module):
    def __init__(self, input_shape=(128, 128, 2), n_channels_out=1, bilinear=True):
        super().__init__()
        n_channels_in = input_shape[2]

        self.inc = inconv(n_channels_in, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(512, 512, 256, bilinear)
        self.up2 = up(256, 256, 128, bilinear)
        self.up3 = up(128, 128, 64, bilinear)
        self.up4 = up(64, 64, 64, bilinear)

        self.outc = outconv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        outputs = self.outc(x)
        return outputs

def build_unet(input_shape=(128, 128, 2), n_channels_out=1, bilinear=True):
    model = UNet(input_shape=input_shape, n_channels_out=n_channels_out, bilinear=bilinear)
    return model