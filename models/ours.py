import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):
    ###################################################
    # Squeeze and Excite Module (Channel Attention)
    ###################################################
    def __init__(self, c, r=16):  # c -> no of channels; r-> reduction ratio
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    
class SpatialAttention(nn.Module):
    ###################################################
    # Spatial Attention
    # uses mean and max of activations
    ###################################################
    def __init__(self, kernel_size=7, out_channels=1):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

class DoubleConv(nn.Module):
    ###################################################
    # Does the following conv operation: 
    # (convolution => [BN] => ReLU) * 2
    ###################################################
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    ###################################################
    # Downscaling with maxpool then double conv
    ###################################################

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    ###################################################
    # Upscaling then double conv
    ###################################################

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module): # output conv layers
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Attention_Guided_UNet(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(Attention_Guided_UNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Defines Unet layers
        ########################################################################
        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        ########################################################################
        # Defines Spatial Attention layers
        ########################################################################
        self.sa1 = SpatialAttention(out_channels=64)
        self.sa2 = SpatialAttention(out_channels=128)
        self.sa3 = SpatialAttention(out_channels=256)
        self.sa4 = SpatialAttention(out_channels=512)
        ########################################################################
         # Defines Channel Attention layers
        ########################################################################
        self.se1 = SE_Block(64)
        self.se2 = SE_Block(128)
        self.se3 = SE_Block(256)
        self.se4 = SE_Block(512)
        ########################################################################

    def count_parameters(self): # Counts number of model parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x): # Forward function
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x4 = x4 * self.sa4(x4)
        x3 = x3 * self.sa3(x3)
        x2 = x2 * self.sa2(x2)
        x1 = x1 * self.sa1(x1)

        x = self.up1(x5, x4)
        x = self.se4(x)
        x = self.up2(x, x3)
        x = self.se3(x)
        x = self.up3(x, x2)
        x = self.se2(x)
        x = self.up4(x, x1)
        x = self.se1(x)
        logits = self.outc(x)
        return F.sigmoid(logits)