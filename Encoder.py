import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torchvision.ops import DeformConv2d


class DeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size
        self.offset = nn.Conv2d(
            in_channels * 2,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1
        )

        self.conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x, ref):
        offset_input = torch.cat([x, ref], dim=1)
        offset = self.offset(offset_input)
        offset = torch.tanh(offset) * 2

        return self.conv(x, offset)




class EdgeEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        edge = torch.abs(self.conv(x))
        return x + edge


class TextureBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


class SaliencyEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context = nn.Conv2d(channels, channels, 5, padding=2)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ctx = self.context(x)
        w = self.attn(ctx)
        return x * w


class GlobalContext(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.pool(
        w = self.fc(w)
        return x + x * w  

class DualModalDualBranchEncoder(nn.Module):

    def __init__(self, in_channels=1, base_channels=64, blocks=2):
        super().__init__()


        self.vis_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.vis_edge = EdgeEnhance(base_channels)
        # self.vis_edge = EdgeEnhance(num_channels=base_channels, growth=64)  

        self.vis_blocks = nn.Sequential(
            *[TextureBlock(base_channels) for _ in range(blocks)]
        )

        self.vis_global = GlobalContext(base_channels)


        self.ir_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.ir_saliency = SaliencyEnhance(base_channels)

        self.ir_blocks = nn.Sequential(
            *[TextureBlock(base_channels) for _ in range(blocks)]
        )

        self.ir_global = GlobalContext(base_channels)

 
        self.down = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.up   = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)

 
        self.vis_out = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.ir_out  = nn.Conv2d(base_channels, base_channels, 3, padding=1)

  
        self.align_dcn = DeformableConv(base_channels, base_channels)

    def forward(self, vis, ir):

        # ========= VIS =========
        vis_feat = self.vis_conv(vis)
        # vis_feat = self.ir_saliency(vis_feat)
        vis_feat = self.vis_edge(vis_feat)
        vis_feat = self.vis_blocks(vis_feat)

        vis_down = self.down(vis_feat)
        vis_up   = self.up(vis_down)


        if vis_up.shape[2:] != vis_feat.shape[2:]:
            vis_up = vis_up[:, :, :vis_feat.shape[2], :vis_feat.shape[3]]

        vis_feat = vis_feat + vis_up

        vis_feat = self.vis_global(vis_feat)
        vis_feat = self.vis_out(vis_feat)
        vis_feat = vis_feat + vis_feat.mean(dim=1, keepdim=True)


        ir_feat = self.ir_conv(ir)
        # ir_feat = self.vis_edge(ir_feat)
        ir_feat = self.ir_saliency(ir_feat)
        ir_feat = self.ir_blocks(ir_feat)

        ir_down = self.down(ir_feat)
        ir_up   = self.up(ir_down)


        if ir_up.shape[2:] != ir_feat.shape[2:]:
            ir_up = ir_up[:, :, :ir_feat.shape[2], :ir_feat.shape[3]]

        ir_feat = ir_feat + ir_up

        ir_feat = self.ir_global(ir_feat)
        ir_feat = self.ir_out(ir_feat)
        ir_feat = ir_feat + ir_feat.mean(dim=1, keepdim=True)


        ir_feat = self.align_dcn(ir_feat, vis_feat)

        return vis_feat, ir_feat

























