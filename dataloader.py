import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from utils import DarkChannel, GuidedFilter
try:
    RESAMPLE_MODE = Image.Resampling.LANCZOS  # Pillow >= 9.1.0
except AttributeError:
    RESAMPLE_MODE = Image.LANCZOS
import glob  # ????????????????
import random
import cv2
import os.path as osp
from torchvision import transforms
from PIL import Image

to_tensor = transforms.Compose([transforms.ToTensor()]) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(1143)  




def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)  
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()  # ????
    return data, filenames


class fusion_dataset_loader(data.Dataset):
    def __init__(self, split,  ir_path=None, vi_path=None):
        super(fusion_dataset_loader, self).__init__()
        self.size = 225  # patch_size
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        if split == 'train':
            data_dir_vis = os.path.join(os.getcwd(), 'train/vi')
            data_dir_ir = os.path.join(os.getcwd(), 'train/ir')
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index] 
            ir_path = self.filepath_ir[index]  

            image_vis = Image.open(vis_path)
            image_vis = image_vis.resize((self.size, self.size), RESAMPLE_MODE) 
            image_vis = np.array(image_vis)
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)


            image_inf = Image.open(ir_path).convert('L')
            image_inf = image_inf.resize((self.size, self.size), RESAMPLE_MODE) 
            image_inf = np.array(image_inf)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)  
            image_vis = torch.tensor(image_vis)
            image_ir = torch.tensor(image_ir)
            return (image_vis, image_ir)

    def __len__(self): 
        return self.length






class fusion_dataset_loader_eval(data.Dataset): 
    def __init__(self, i, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path 
            # elif sub_dir == 'vi_en':  
            elif sub_dir == 'vi':  
                # self.vis_path = osp.join(temp_path, str(i + 1))
                 self.vis_path = temp_path  

        self.name_list = os.listdir(self.inf_path)  
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index] 
        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  
        vis_image = Image.open(os.path.join(self.vis_path, name))  
        ir_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        return vis_image, ir_image, name

    def __len__(self):
        return len(self.name_list)





class fusion_dataset_loader_test(data.Dataset): 
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path 
            elif sub_dir == 'vi':
                self.vis_path = osp.join(temp_path)
        self.name_list = os.listdir(self.inf_path)  
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  
        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  
        vis_image = Image.open(os.path.join(self.vis_path, name))
        ir_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        return vis_image, ir_image, name

    def __len__(self):
        return len(self.name_list)


def rgb2ycbcr(input_im): 
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = torch.clamp(Y, min=0., max=1.0)
    Cr = torch.clamp(Cr, min=0., max=1.0)
    Cb = torch.clamp(Cb, min=0., max=1.0)
    Y = torch.unsqueeze(Y, 1)  # ????????
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    # temp = torch.cat((Y, Cr, Cb), dim=1) CPU????
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (temp.reshape(
        list(input_im.size())[0],
        list(input_im.size())[2],
        list(input_im.size())[3],
        3, )
           .transpose(1, 3)
           .transpose(2, 3))
    return out


def ycbcr2rgb(input_im): 
    B, C, W, H = input_im.shape
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]])
    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3).cuda()
    out = torch.clamp(out, min=0., max=1.0)
    return out


