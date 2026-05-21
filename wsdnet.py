import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import math


class PSAF(nn.Module):
    def __init__(self, dim, prior_dim):
        super().__init__()
        self.cross_attention = CrossAttention(dim)
        self.csaf = CSAF(dim, prior_dim)
        self.fusion = Fusion(dim)
        # self.spatial_attn = SpatialAttention(dim)

    def forward(self, feat_A, feat_B, prior_A, prior_B):

        feat_A, feat_B = self.cross_attention(feat_A, feat_B)
        feat_A = self.csaf(feat_A, prior_A)
        feat_B = self.csaf(feat_B, prior_B)

        # 3. fusion
        # out = self.fusion(feat_A, feat_B)



        score_A = torch.mean(torch.abs(feat_A), dim=1, keepdim=True)
        score_B = torch.mean(torch.abs(feat_B), dim=1, keepdim=True)

        weights = torch.softmax(torch.cat([score_A, score_B], dim=1), dim=1)

        out = weights[:, 0:1] * feat_A + weights[:, 1:2] * feat_B

        return out





        # 4. spatial attention
        # out = self.spatial_attn(fused)

        # return out




class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, A, B):
        B_, C, H, W = A.shape

        qA = self.q(A).view(B_, C, -1)              # B,C,N
        kB = self.k(B).view(B_, C, -1)              # B,C,N
        vB = self.v(B).view(B_, C, -1)

        attn = torch.softmax(torch.bmm(qA, kB.transpose(1, 2)) / (C ** 0.5), dim=-1)

        out_A = torch.bmm(attn, vB).view(B_, C, H, W)
        out_A = A + self.proj(out_A)

        qB = self.q(B).view(B_, C, -1)
        kA = self.k(A).view(B_, C, -1)
        vA = self.v(A).view(B_, C, -1)

        attn2 = torch.softmax(torch.bmm(qB, kA.transpose(1, 2)) / (C ** 0.5), dim=-1)
        out_B = torch.bmm(attn2, vA).view(B_, C, H, W)
        out_B = B + self.proj(out_B)

        return out_A, out_B




class CSAF(nn.Module):
    def __init__(self, feat_dim, prior_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(feat_dim, affine=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_dim, feat_dim // 2, 3, padding=1),
            nn.ReLU()
        )

        self.gamma = nn.Conv2d(feat_dim // 2, feat_dim, 3, padding=1)
        self.beta  = nn.Conv2d(feat_dim // 2, feat_dim, 3, padding=1)

    def forward(self, x, prior):
        prior = F.interpolate(prior, size=x.shape[2:], mode='bilinear')

        h = self.mlp_shared(prior)
        gamma = torch.sigmoid(self.gamma(h))
        beta  = self.beta(h)

        x = self.norm(x)
        return x * (1 + gamma) + beta



class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, A, B):
        return self.conv(torch.cat([A, B], dim=1))


# def _calculate_detail_weight(y, use_structure_tensor=True):
#     if y.shape[1] > 1:
#         y = torch.mean(y, dim=1, keepdim=True)

#     if use_structure_tensor:
#         scharr_x = torch.tensor([[-3., 0., 3.],
#                                  [-10., 0., 10.],
#                                  [-3., 0., 3.]], device=y.device).view(1,1,3,3)
#         scharr_y = torch.tensor([[-3., -10., -3.],
#                                  [0., 0., 0.],
#                                  [3., 10., 3.]], device=y.device).view(1,1,3,3)

#         grad_x = F.conv2d(y, scharr_x, padding=1)
#         grad_y = F.conv2d(y, scharr_y, padding=1)
#         base = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
#     else:
#         sobel_x = torch.tensor([[-1., 0., 1.],
#                                 [-2., 0., 2.],
#                                 [-1., 0., 1.]], device=y.device).view(1,1,3,3)
#         sobel_y = torch.tensor([[-1., -2., -1.],
#                                 [0., 0., 0.],
#                                 [1., 2., 1.]], device=y.device).view(1,1,3,3)

#         grad_x = F.conv2d(y, sobel_x, padding=1)
#         grad_y = F.conv2d(y, sobel_y, padding=1)
#         base = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

#     w1 = base
#     w2 = F.interpolate(base, scale_factor=0.5, mode='bilinear', align_corners=False)
#     w3 = F.interpolate(base, scale_factor=0.25, mode='bilinear', align_corners=False)

#     w2 = F.interpolate(w2, size=y.shape[2:], mode='bilinear', align_corners=False)
#     w3 = F.interpolate(w3, size=y.shape[2:], mode='bilinear', align_corners=False)

#     W = (w1 + 0.8*w2 + 0.6*w3) / 2.4
#     W = (W - W.min()) / (W.max() - W.min() + 1e-8)

#     return W


def _calculate_detail_weight(y, use_structure_tensor=True):
    # ====== ?? ?????????????????????? ======
    if not hasattr(_calculate_detail_weight, "alpha"):
        _calculate_detail_weight.alpha = nn.Parameter(
            torch.tensor([1.0, 0.8, 0.6], device=y.device)
        )

    alpha = _calculate_detail_weight.alpha

    # ====== ?????? ======
    if y.shape[1] > 1:
        y = torch.mean(y, dim=1, keepdim=True)

    if use_structure_tensor:
        scharr_x = torch.tensor([[-3., 0., 3.],
                                 [-10., 0., 10.],
                                 [-3., 0., 3.]], device=y.device).view(1,1,3,3)
        scharr_y = torch.tensor([[-3., -10., -3.],
                                 [0., 0., 0.],
                                 [3., 10., 3.]], device=y.device).view(1,1,3,3)
    else:
        scharr_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]], device=y.device).view(1,1,3,3)
        scharr_y = torch.tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]], device=y.device).view(1,1,3,3)

    grad_x = F.conv2d(y, scharr_x, padding=1)
    grad_y = F.conv2d(y, scharr_y, padding=1)
    base = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    w1 = base
    w2 = F.interpolate(base, scale_factor=0.5, mode='bilinear', align_corners=False)
    w3 = F.interpolate(base, scale_factor=0.25, mode='bilinear', align_corners=False)

    w2 = F.interpolate(w2, size=y.shape[2:], mode='bilinear', align_corners=False)
    w3 = F.interpolate(w3, size=y.shape[2:], mode='bilinear', align_corners=False)

    weights = torch.softmax(alpha, dim=0)

    W = (
        weights[0] * w1 +
        weights[1] * w2 +
        weights[2] * w3
    )

    W = (W - W.min()) / (W.max() - W.min() + 1e-8)

    return W



class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # depthwise
            nn.Conv2d(dim, dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn + x



class WTLowFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1x1 = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.conv_low = nn.Conv2d(channels, channels, 3, padding=1)
        

        self.alpha_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 4, 1), 
            nn.Softmax(dim=1)
        )
        

        self.base_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.register_buffer("haar_kernels", self._build_haar_kernels())
    
    def _build_haar_kernels(self):
        LL = torch.tensor([[1, 1, 0],
                           [1, 1, 0],
                           [0, 0, 0]], dtype=torch.float32)
        
        LH = torch.tensor([[-1, -1, 0],
                           [1,  1, 0],
                           [0,  0, 0]], dtype=torch.float32)
        
        HL = torch.tensor([[-1, 1, 0],
                           [-1, 1, 0],
                           [0,  0, 0]], dtype=torch.float32)
        
        HH = torch.tensor([[1, -1, 0],
                           [-1, 1, 0],
                           [0,  0, 0]], dtype=torch.float32)
        
        kernels = torch.stack([LL, LH, HL, HH], dim=0)  # [4, 3, 3]
        kernels = kernels.unsqueeze(1)  # [4, 1, 3, 3]
        kernels = kernels / 4.0
        
        return kernels  # [4, 1, 3, 3]
    
    def dynamic_wavelet_conv(self, x):
        B, C, H, W = x.shape
        alpha = self.alpha_net(x)  # [B, 4, 1, 1]
        W_base = self.base_conv.weight  # [C_out, C_in, 3, 3] = [C, C, 3, 3]
        haar_kernels = self.haar_kernels  # [4, 1, 3, 3]
        # [4, 1, 3, 3] -> [4, 1, 1, 3, 3] 
        haar_expanded = haar_kernels.unsqueeze(1)  # [4, 1, 1, 3, 3]
        # [C, C, 3, 3] -> [1, C, C, 3, 3] -> [4, C, C, 3, 3]
        W_base_expanded = W_base.unsqueeze(0)  # [1, C, C, 3, 3]
        W_base_expanded = W_base_expanded.expand(4, C, C, 3, 3)  # [4, C, C, 3, 3]
        # haar_expanded: [4, 1, 1, 3, 3]
        # W_base_expanded: [4, C, C, 3, 3]
        W_modulated = W_base_expanded * haar_expanded  # [4, C, C, 3, 3]
        # alpha: [B, 4, 1, 1] -> [B, 4, 1, 1, 1, 1]
        alpha_expanded = alpha.view(B, 4, 1, 1, 1, 1)
        # W_modulated: [4, C, C, 3, 3] -> [1, 4, C, C, 3, 3]
        W_modulated = W_modulated.unsqueeze(0)  # [1, 4, C, C, 3, 3]
        W_dyn = (alpha_expanded * W_modulated).sum(dim=1)  # [B, C, C, 3, 3]
        x_group = x.view(1, B * C, H, W)  # [1, B*C, H, W]
        W_dyn_reshaped = W_dyn.view(B * C, C, 3, 3)  # [B*C, C, 3, 3]
        
        out = F.conv2d(
            x_group, 
            W_dyn_reshaped, 
            padding=1, 
            groups=B
        )  # [1, B*C, H, W]
        
        out = out.view(B, C, H, W)  # [B, C, H, W]
        
        return out
    
    def forward(self, vis_LL, ir_LL):
        """
            vis_LL: [B, C, H, W] 
            ir_LL: [B, C, H, W] 
            LL_enhanced: [B, C, H, W] 
        """
       ========= Step1: Max-Min =========
        max_v = torch.maximum(vis_LL, ir_LL)
        min_v = torch.minimum(vis_LL, ir_LL)
        contrast = max_v - min_v
        
        fused_feat = torch.cat([max_v, min_v, contrast], dim=1)
        fused_feat = torch.sigmoid(self.conv1x1(fused_feat))
      
        gamma = 5
        fused = contrast + gamma * fused_feat
        
        LL1_A = self.conv_low(fused)
        LL1_B = self.dynamic_wavelet_conv(fused)
        LL_enhanced = LL1_A + LL1_B
        
        return LL_enhanced






    
class WaveletFusionModule(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
        self.iwt = DWTInverse(mode='zero', wave='haar').cuda()
 
        ################################
        # Low frequency attention
        ################################

        self.low_att = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

        ################################
        # High frequency attention
        ################################

        self.high_att = nn.Sequential(
            nn.Conv2d(channels * 6, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Sigmoid()
        )

        self.psaf_hl = PSAF(dim=channels, prior_dim=1) 
        self.psaf_lh = PSAF(dim=channels, prior_dim=1)  
        self.psaf_hh = PSAF(dim=channels, prior_dim=1)  

        self.low_fusion = WTLowFusion(channels)   
        



        

    def forward(self, vis_img,ir_img,vis_feat, ir_feat):

      

        vis_LL, vis_H = self.dwt(vis_feat)
        ir_LL, ir_H = self.dwt(ir_feat)


    

        low_fused = self.low_fusion(vis_LL, ir_LL)

        vis_H = vis_H[0]
        ir_H  = ir_H[0]

        vis_HL = vis_H[:, :, 0]
        vis_LH = vis_H[:, :, 1]
        vis_HH = vis_H[:, :, 2]

        ir_HL = ir_H[:, :, 0]
        ir_LH = ir_H[:, :, 1]
        ir_HH = ir_H[:, :, 2]


        vis_prior = _calculate_detail_weight(vis_img)
        ir_prior  = _calculate_detail_weight(ir_img)

        # HL 
        fused_HL =  self.psaf_hl(vis_HL, ir_HL, vis_prior, ir_prior)
        # LH 
        fused_LH =  self.psaf_lh(vis_LH, ir_LH, vis_prior, ir_prior)
        # HH
        fused_HH =  self.psaf_hh(vis_HH, ir_HH, vis_prior, ir_prior)
        fused_H = torch.stack([fused_HL, fused_LH, fused_HH], dim=2)



        ################################
        # Inverse wavelet
        ################################

        fused = self.iwt((low_fused, [fused_H]))

        ################################
        # size alignment
        ################################

        if fused.shape[2:] != vis_feat.shape[2:]:

            fused = fused[:, :, :vis_feat.shape[2], :vis_feat.shape[3]]

        ################################
        # residual
        ################################

        fused = fused + 0.6 * vis_feat + 0.4 * ir_feat

        return fused







class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w  




class ReconstructionModule(nn.Module):
    def __init__(self, channels, out_channels=1):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x):

        feat = self.body(x)
        return self.out(feat)


class FeatureRefinement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x):
        return x + self.refine(x)  # ????????


###############################################
# Fusion Network
###############################################

class MultiBranchFusionNet(nn.Module):

    def __init__(self, in_channels=64, feat_channels=32, out_channels=1):

        super().__init__()

        ################################
        # Frequency Fusion
        ################################

        self.wavelet_fusion = WaveletFusionModule(in_channels)

        ################################
        # Feature Compression
        ################################

        self.conv1 = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        self.conv2 = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)

        ################################
        # Refinement
        ################################

        self.refine = FeatureRefinement(feat_channels)

        ################################
        # Reconstruction
        ################################

        self.reconstruction = ReconstructionModule(
            feat_channels,
            out_channels
        )

    def forward(self, vis_img, ir_img,vis_feat, ir_feat):

        ################################
        # Frequency Fusion
        ################################

        fused_feat = self.wavelet_fusion(vis_img, ir_img,vis_feat, ir_feat)

        # ################################
        # # Feature Compression
        # ################################

        fused_feat = F.relu(self.conv1(fused_feat))
        fused_feat = F.relu(self.conv2(fused_feat))

        # ################################
        # # Feature Refinement
        # ################################

        fused_feat_refine = self.refine(fused_feat)

        ################################
        # Reconstruction
        ################################
 
        output = self.reconstruction(fused_feat_refine)

        return output



















