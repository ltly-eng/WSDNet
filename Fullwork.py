import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import DualModalDualBranchEncoder
from wsdnet import MultiBranchFusionNet


class FullFusionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = DualModalDualBranchEncoder(
            in_channels=1,
            base_channels=64
        )
        self.fusion_net = MultiBranchFusionNet(
            in_channels=64,
            feat_channels=16,
            out_channels=1
        )


    def forward(self, vis_img, ir_img):
        vis_feat, ir_feat = self.encoder(vis_img, ir_img)
        fused_img = self.fusion_net(vis_img, ir_img,vis_feat, ir_feat)
        if fused_img.shape[2:] != vis_img.shape[2:]:
            fused_img = F.interpolate(
                fused_img,
                size=vis_img.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        return fused_img









