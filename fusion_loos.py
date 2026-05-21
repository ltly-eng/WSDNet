import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import rgb2ycbcr,ycbcr2rgb
from math import exp





class fusionloss(nn.Module):
    def __init__(self):
        super(fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.mse_loss = nn.MSELoss()  
        self.fuionnet = model.FusionNet().cuda()
        self.angle=angle() 



       
    def forward(self,vi_y,vi,ir,y_f,I_f):
        vi_y_gard=self.sobelconv(vi_y)
        ir_gard=self.sobelconv(ir)
        y_f_grad=self.sobelconv(y_f)

        max_grad=torch.max(vi_y_gard,ir_gard)
        grad_loss = F.l1_loss(max_grad,y_f_grad)
        max_init = torch.max(ir,vi_y)
        image_loss = F.l1_loss(y_f, max_init)
        total_loss = 2*image_loss + 2*grad_loss +
        return total_loss,image_loss,grad_loss



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
def get_per(img):
	fro_2_norm = torch.sum(torch.pow(img,2),dim=[1,2,3]) # ????Frobenius????
	loss=fro_2_norm / (225.0*225.0) # ??????
	return loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()





def final_ssim(img_ir, img_vis, img_fuse):
    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)
    std_ir = std(img_ir)
    std_vi = std(img_vis)
    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)
    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one) 
    ssim = map1 * ssim_ir + map2 * ssim_vi
    return ssim.mean()
