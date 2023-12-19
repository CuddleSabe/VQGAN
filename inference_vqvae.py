import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.utils import  img2tensor
from torch.utils import data as data
import cv2
from basicsr.utils.img_util import tensor2img
from VQGAN.archs.vqvae_arch import VQVAE, VQVAE_multi_codebook
from basicsr.utils import  img2tensor
import argparse
from torch.nn import functional as F


def pad_test(lq,scale):
    if scale==1:
        window_size = 32
    elif scale==2:
        window_size = 16
    else:
        window_size = 8      
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = lq.size()
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    lq = F.pad(lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return lq,mod_pad_h,mod_pad_w

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=1)
    
    parser.add_argument('--model_path', type=str, default='/root/picasso/韩昊天/画质修复/VQGAN/experiments/Degrade/models/net_g_80000.pth')
    parser.add_argument('--im_path', type=str, default='/root/picasso/韩昊天/画质修复/VQGAN/face_degrade')
    
    parser.add_argument('--res_path', type=str, default='./test/degrade_face')
    args = parser.parse_args()

    os.makedirs(args.res_path, exist_ok=True)
    # model = VQVAE(input_dim=48, dim=64, n_embedding=256)
    model = VQVAE_multi_codebook(input_dim=48, dim=64, n_embedding=256, n_codebook=64)
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params_ema'], strict=False)

    model.to('cuda:0')
    model.eval()

    im_list = os.listdir(args.im_path)
    im_list.sort()
    im_list = [name for name in im_list if (name.endswith('.jpeg') or name.endswith('.png') or name.endswith('.jpg'))]

    ze_last = None
    with torch.no_grad():
        z = None
        for name in sorted(im_list):
            # if int(name.split('.')[0]) < 19:
            #     continue
            path = os.path.join(args.im_path, name)
            im_np = cv2.imread(path)
            im = img2tensor(im_np)
            im = im.unsqueeze(0).cuda()/255.
            # im = torch.nn.functional.interpolate(im, scale_factor=4)
            lq,mod_pad_h,mod_pad_w= pad_test(im,args.scale)
            timer = time.time()
            sr, _, _ = model(lq)
            torch.cuda.synchronize()
            timer = (time.time() - timer) * 1000
            _, _, h, w = sr.size()
            sr = sr[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]
            im_sr = tensor2img(sr, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
            save_path = os.path.join(args.res_path, name.split('.')[0]+'.png')
            im_sr = np.concatenate((im_np, im_sr), axis=1)
            cv2.imwrite(save_path, im_sr)
            print(save_path, w, 'x', h, 'time:', timer)
    
