import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#????GPU????
import numpy as np
from torch.autograd import Variable
import argparse
import datetime
import time
import math
import logging
import os.path as osp
import torch
import dataloader
from fusion_loss import fusionloss,final_ssim
from logger import setup_logger
from torch.utils.data import DataLoader
from dataloader import rgb2ycbcr,ycbcr2rgb
from torchvision import transforms
from tqdm import tqdm
from thop import profile
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torchvision import transforms
import cv2
import Fullwork   
    

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)

        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)

    elif classname.find('conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)


def EN_function(image_array):
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 255))
    histogram = histogram / np.sum(histogram)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy

def SF_function(image):
    RF = np.diff(image, axis=0)
    CF = np.diff(image, axis=1)
    SF = np.sqrt(np.mean(RF ** 2) + np.mean(CF ** 2))
    return SF

def SD_function(image_array):
    return np.std(image_array)

def PSNR_function(ir_img, vi_img, f_img):
    ir_img = ir_img.astype(np.float64)
    vi_img = vi_img.astype(np.float64)
    f_img = f_img.astype(np.float64)
    mse_ir = np.mean(((f_img - ir_img) / 255.0) ** 2)
    mse_vi = np.mean(((f_img - vi_img) / 255.0) ** 2)
    mse = (mse_ir + mse_vi) / 2
    psnr = 20 * np.log10(255.0 / np.sqrt(mse + 1e-7))
    return psnr

def AG_function(image):
    gradx, grady = np.gradient(image)
    s = np.sqrt((gradx ** 2 + grady ** 2) / 2)
    return np.mean(s)

def MI_function(A, B, F, gray_level=256):
    def joint_histogram(im1, im2):
        range = [[0, gray_level-1], [0, gray_level-1]]
        return np.histogram2d(im1.ravel(), im2.ravel(), bins=gray_level, range=range)[0] / (im1.size)
    
    def marginal_histogram(im):
        return np.histogram(im.ravel(), bins=gray_level, range=(0, gray_level-1))[0] / (im.size)
    
    h_AF = joint_histogram(A, F)
    h_BF = joint_histogram(B, F)
    
    P_A = marginal_histogram(A)
    P_B = marginal_histogram(B)
    P_F = marginal_histogram(F)
    
    MI_AF = np.sum(h_AF * np.log2((h_AF + 1e-7) / (P_A[:, None] * P_F[None, :] + 1e-7)))
    MI_BF = np.sum(h_BF * np.log2((h_BF + 1e-7) / (P_B[:, None] * P_F[None, :] + 1e-7)))
    
    return MI_AF + MI_BF


def CC_function(A, B, F):
    rAF = np.corrcoef(A.ravel(), F.ravel())[0, 1]
    rBF = np.corrcoef(B.ravel(), F.ravel())[0, 1]
    return np.mean([rAF, rBF])



def VIF_function(img1, img2, img_fusion):
    def vifp_mscale(ref, dist):
        ref = (ref * 255.0).astype(np.float64)
        dist = (dist * 255.0).astype(np.float64)

        sigma_nsq = 2.0
        eps = 1e-10

        num = 0.0
        den = 0.0

        for scale in range(4):
            N = 2 ** (4 - scale) + 1
            sd = N / 5.0

            mu1 = cv2.GaussianBlur(ref, (N, N), sd)
            mu2 = cv2.GaussianBlur(dist, (N, N), sd)

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = cv2.GaussianBlur(ref * ref, (N, N), sd) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(dist * dist, (N, N), sd) - mu2_sq
            sigma12 = cv2.GaussianBlur(ref * dist, (N, N), sd) - mu1_mu2

            sigma1_sq = np.maximum(0, sigma1_sq)
            sigma2_sq = np.maximum(0, sigma2_sq)

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq = np.maximum(sv_sq, eps)

            num += np.sum(np.log(1.0 + (g ** 2) * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log(1.0 + sigma1_sq / sigma_nsq))

            if scale < 3:
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

        return num / (den + eps)

    vif_vis = vifp_mscale(img1, img_fusion)
    vif_ir = vifp_mscale(img2, img_fusion)

    return (vif_vis + vif_ir) / 2


def Qabf_function(rgb_vis, rgb_ir, rgb_fusion):

    def to_255(img):
        if img.max() <= 1.0:
            img = img * 255.0
        return img.astype(np.float32)
    
    vis = to_255(rgb_vis)
    ir = to_255(rgb_ir)
    fusion = to_255(rgb_fusion)

    def rgb2gray(img):
        if img.shape[0] == 1:
            return img[0]
        else:
            return cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_RGB2GRAY).astype(np.float32)

    vis_gray = rgb2gray(vis)
    ir_gray = rgb2gray(ir)
    fusion_gray = rgb2gray(fusion)

    def gradient(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return mag

    grad_vis = gradient(vis_gray)
    grad_ir = gradient(ir_gray)
    grad_fusion = gradient(fusion_gray)
    eps = 1e-10
    G_vis = np.sum(np.minimum(grad_vis, grad_fusion)) / (np.sum(grad_vis) + eps)
    G_ir = np.sum(np.minimum(grad_ir, grad_fusion)) / (np.sum(grad_ir) + eps)

    L_vis = 1 - np.mean(np.abs(vis_gray - fusion_gray)) / 255.0
    L_ir = 1 - np.mean(np.abs(ir_gray - fusion_gray)) / 255.0

    Qabf_value = 0.5 * (G_vis * L_vis + G_ir * L_ir)

    # ?????? [0,1]
    Qabf_value = np.clip(Qabf_value, 0.0, 1.0)

    return float(Qabf_value)


def SSIM_function(img1, img2):
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)
    
    return ssim(img1_uint8, img2_uint8, data_range=255)



def SCD_function(ir_img, vi_img, f_img):
    ir_img = ir_img.astype(np.float64)
    vi_img = vi_img.astype(np.float64)
    f_img = f_img.astype(np.float64)
    ir_vi_diff = np.abs(ir_img - vi_img)
    ir_vi_sum = np.sum(ir_img) + np.sum(vi_img)
    ir_vi_scd = np.sum(ir_vi_diff) / (ir_vi_sum + 1e-6)

    f_ir_diff = np.abs(f_img - ir_img)
    f_vi_diff = np.abs(f_img - vi_img)
    f_ir_sum = np.sum(f_img) + np.sum(ir_img)
    f_vi_sum = np.sum(f_img) + np.sum(vi_img)

    f_ir_scd = np.sum(f_ir_diff) / (f_ir_sum + 1e-6)
    f_vi_scd = np.sum(f_vi_diff) / (f_vi_sum + 1e-6)
    scd = (f_ir_scd + f_vi_scd) / (ir_vi_scd + 1e-6)

    return scd





def train_fusion(i, logger=None): 
    modelpth = './model'
    modelpth = osp.join(modelpth, str(i+1))  
    os.makedirs(modelpth, mode=0o777, exist_ok=True)
    
    try:try 
        fusion_batch_size = 5 
        n_workers = 4 
        
        ds = dataloader.fusion_dataset_loader('train')  
        dl = torch.utils.data.DataLoader(
            ds, 
            batch_size=fusion_batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=False
        )
        # dl????????????vi_en??ir

        # net = model.FusionNet()  

        net = Fullwork.FullFusionModel()

        if i == 0:  
            net.apply(weights_init)
        if i > 0: 
            load_path = './model'
            load_path = osp.join(load_path, str(i), 'fusion_model.pth')
            net.load_state_dict(torch.load(load_path))
            print('Load Pre-trained Fusion Model:{}!'.format(load_path)) 
        net.cuda()
        net.eval()
        net.train()

        # enhancemodel = model.luminance_adjustment().cuda()  
        lr_start = 2*1e-5
        optim = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=0.0001)
        




        criteria_fusion = fusionloss() 
        st = glob_st = time.time()
        epoch = 20 
        grad_step = 5.0  
        dl.n_iter = len(dl) 

        for epo in range(0, epoch):

            lr_decay = 0.75
            lr_this_epo = lr_start * lr_decay ** ((epo / 5) + 1) 
            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_epo
            
            for it, (image_vis, image_ir) in enumerate(dl):
                net.train()
                image_vis = Variable(image_vis, requires_grad=True).cuda()
                image_ir = Variable(image_ir, requires_grad=True).cuda()
                # _, image_vis_en, _ = enhancemodel(image_vis) 
                image_vis_ycbcr = rgb2ycbcr(image_vis)  
                vi_y = image_vis_ycbcr[:, 0:1, :, :]
        
                Y_f = net(vi_y, image_ir)


              


             
                fusion_ycbcr = torch.cat(
                    (Y_f, image_vis_ycbcr[:, 1:2, :, :], image_vis_ycbcr[:, 2:, :, :]),
                    dim=1)
           
                I_f = ycbcr2rgb(fusion_ycbcr)

              
                ones = torch.ones_like(I_f)  
                zeros = torch.zeros_like(I_f) 
                I_f = torch.where(I_f > ones, ones, I_f) 
                I_f = torch.where(I_f < zeros, zeros, I_f)  



       




                loss_fusion, loss_image, loss_grad, loss_color= criteria_fusion(vi_y, image_vis, image_ir, Y_f, I_f)
                ssim_loss = 0
                ssim_loss_temp = 1 - final_ssim(image_ir, vi_y, Y_f)
                ssim_loss += ssim_loss_temp
                ssim_loss /= len(Y_f)
                loss_fusion = loss_fusion + 50*ssim_loss  # ????SSIM????
                loss_fusion.backward()


                if grad_step > 1:
                    loss_fusion = loss_fusion / grad_step 
                
                # ??????????????
                if (it + 1) % grad_step == 0:
                    optim.step()
                    optim.zero_grad()
                
                # ??????????????????
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = dl.n_iter * epo + it + 1
                
                if (it + 1) % 50 == 0:
                    lr = optim.param_groups[0]['lr']
                    eta = int((dl.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
                    eta = str(datetime.timedelta(seconds=eta))
                    msg = ', '.join(
                        ['step: {it}/{max_it}', 
                         'loss_fusion:{loss_fusion:.4f}\n',  
                         'loss_image: {loss_image:.4f}', 
                         'loss_grad: {loss_grad:4f}',  
                         'loss_color: {loss_color:4f}', 
                         'loss_ssim:{loss_ssim:4f}', 
                         'eta: {eta}',
                         'time: {time:.4f}', ]).format( 
                        it=now_it, max_it=dl.n_iter * epoch, lr=lr,
                        loss_fusion=loss_fusion, loss_image=loss_image,
                        loss_grad=loss_grad, loss_color=loss_color,
                        loss_ssim=ssim_loss, eta=eta, time=t_intv, )
                    logger.info(msg)
                    st = ed
        
        save_pth = osp.join(modelpth, 'fusion_model.pth') 
        state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        torch.save(state, save_pth)  # ????????????
        logger.info('Fusion Model Training done~, The Model is saved to: {}'.format(save_pth))  
        logger.info('\n') 








def run_fusion(i): 
    fusion_model_path = osp.join(os.getcwd(), 'model', str(i + 1), 'fusion_model.pth')
    fusion_dir = osp.join(os.getcwd(), 'eval/If', str(i + 1))
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)

    fusionmodel = Fullwork.FullFusionModel().cuda()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel loaded!')
    testdataset = dataloader.fusion_dataset_loader_eval(i, osp.join(os.getcwd(), 'eval'))


    testloader = DataLoader(
        dataset=testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    testtqdm = tqdm(testloader, total=len(testloader))

    metrics_list = []

    with torch.no_grad():
        for images_vis, images_ir, name in testtqdm:
            images_vis = images_vis.cuda()
            images_ir = images_ir.cuda()

            image_vis_ycbcr = rgb2ycbcr(images_vis)
            image_vis_y = image_vis_ycbcr[:, 0:1, :, :]

            Y_f = fusionmodel(image_vis_y, images_ir)
            fusion_ycbcr = torch.cat((Y_f, image_vis_ycbcr[:, 1:2, :, :], image_vis_ycbcr[:, 2:, :, :]), dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)
            I_f = torch.clamp(I_f, 0.0, 1.0).cpu().numpy()  # [B,C,H,W], float[0,1]

            for k in range(len(name)):
                image_I_f = I_f[k]       # [C,H,W]
                image_vis_k = images_vis[k].cpu().numpy()
                image_ir_k = images_ir[k].cpu().numpy()

                image_I_f_t = torch.tensor(image_I_f).to(images_vis.device)
                image_I_f_pil = transforms.ToPILImage()(image_I_f_t)
                save_path = osp.join(fusion_dir, name[k])
                image_I_f_pil.save(save_path)
                def to_gray(img):
                    if img.max() <= 1.0:
                        img = img * 255.0
                    img = img.astype(np.float32)
                    if img.shape[0] == 1:
                        return img[0]
                    else:
                        rgb = img.transpose(1, 2, 0)
                        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

                fusion_gray = to_gray(image_I_f)
                vis_gray = to_gray(image_vis_k)
                ir_gray = to_gray(image_ir_k)
                EN = EN_function(fusion_gray)
                SF = SF_function(fusion_gray)
                SD = SD_function(fusion_gray)
                AG = AG_function(fusion_gray)
                MI = MI_function(vis_gray, ir_gray, fusion_gray)
                PSNR = PSNR_function(ir_gray, vis_gray, fusion_gray)
                SCD = SCD_function(ir_gray, vis_gray, fusion_gray)  # ????????
                CC = CC_function(vis_gray, ir_gray, fusion_gray) # ????????
                fusion_gray_norm = fusion_gray / 255.0
                vis_gray_norm = vis_gray / 255.0
                ir_gray_norm = ir_gray / 255.0
                SSIM_vis = SSIM_function(vis_gray_norm, fusion_gray_norm)
                SSIM_ir = SSIM_function(ir_gray_norm, fusion_gray_norm)
                SSIM = (SSIM_vis + SSIM_ir) / 2  # ????????
                fusion_gray_norm = fusion_gray / 255.0
                vis_gray_norm = vis_gray / 255.0
                ir_gray_norm = ir_gray / 255.0
                VIF = VIF_function(vis_gray_norm, ir_gray_norm, fusion_gray_norm)
                Qabf = Qabf_function(image_vis_k, image_ir_k, image_I_f)

                metrics_list.append({
                    'Name': name[k],
                    'EN': EN,
                    'SF': SF,
                    'SD': SD,
                    'AG': AG,
                    'MI': MI,
                    'PSNR': PSNR,
                    'SCD': SCD,  # ????????
                    'CC': CC,  # ????SSIM????
                    'SSIM': SSIM,  # ????SSIM????
                    'VIF': VIF,
                    'Qabf': Qabf
                    
                })

        df = pd.DataFrame(metrics_list)
        metric_cols = ['EN', 'SF', 'SD', 'AG', 'MI', 'PSNR', 'SCD','CC','SSIM', 'VIF', 'Qabf']


        mean_values = df[metric_cols].mean()
        std_values = df[metric_cols].std()

        mean_std_row = [''] + [
            f"{mean_values[col]:.4f} ? {std_values[col]:.4f}" for col in metric_cols
        ]

        mean_std_df = pd.DataFrame([mean_std_row], columns=df.columns)
        df = pd.concat([df, mean_std_df], ignore_index=True)

        save_excel_path = osp.join(fusion_dir, f"fusion_metrics_{i + 1}.xlsx")
        df.to_excel(save_excel_path, index=False)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    # parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()

   
    # ??????????
    logpath='./logs' # ????????
    logger = logging.getLogger() # ????logger
    setup_logger(logpath) # ????????

   
    # ????????????5????????
    for i in range (0,5):
        torch.cuda.empty_cache()
        print(f"\n=== Iteration {i+1} Start ===")
        print("Initial Memory:", torch.cuda.memory_summary(device=0, abbreviated=False))
        train_fusion(i, logger)  # RFN train(????????????)
        print("\nAfter train_fusion:")
        print(torch.cuda.memory_summary(device=0, abbreviated=False))
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        torch.cuda.empty_cache()  # ??????????????
        print("\nBefore run_fusion:")
        print(torch.cuda.memory_summary(device=0, abbreviated=False))
        run_fusion(i)  # RFN eval(????????????)
        print("\nAfter run_fusion:")
        print(torch.cuda.memory_summary(device=0, abbreviated=False))
        print("|??????|{0} Fusion Image Sucessfully~!".format(i + 1))
        print("??????????????????????????????????????????????????????")
    print("Training Done!")

