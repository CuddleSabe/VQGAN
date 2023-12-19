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
from VQGAN.archs.pixelcnn_arch import PixelCNNWithEmbedding
from basicsr.utils import  img2tensor
import argparse
from torch.nn import functional as F
import einops
from tqdm import tqdm


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
    
    parser.add_argument('--vae_path', type=str, default='/root/picasso/韩昊天/画质修复/VQGAN/experiments/VQVAE_Multi_FFHQ_codebook8/models/net_g_latest.pth')
    parser.add_argument('--model_path', type=str, default='/root/picasso/韩昊天/画质修复/VQGAN/experiments/pixelcnn_multi_8/models/net_g_180000.pth')
    
    parser.add_argument('--res_path', type=str, default='./test/pixelcnn_vae_multi8')
    args = parser.parse_args()

    os.makedirs(args.res_path, exist_ok=True)
    
    vae = VQVAE_multi_codebook(input_dim=48, dim=64, n_embedding=256, n_codebook=8)
    # vae = VQVAE(input_dim=48, dim=64, n_embedding=256)
    loadnet = torch.load(args.vae_path, map_location=torch.device('cpu'))
    vae.load_state_dict(loadnet['params_ema'], strict=False)
    vae.to('cuda:0')
    vae.eval()
    
    model = PixelCNNWithEmbedding(n_blocks=35, p=256, linear_dim=256, bn=True, color_level=256)
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params_ema'], strict=False)
    model.to('cuda:0')
    model.eval()

    n_sample = 1
    input_shape = (n_sample, 16*8, 16)
    x = torch.zeros(input_shape).cuda().to(torch.long)
    
    
    with torch.no_grad():
        for i in tqdm(range(input_shape[1])):
            for j in range(input_shape[2]):
                output = model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1)
                x[:, i, j] = pixel[:, 0]
                
    imgs = vae.decode_idx(x)
    imgs = imgs * 255
    imgs = imgs.clip(0, 255)
    imgs = einops.rearrange(imgs,
                            '(n1 n2) c h w -> (n1 h) (n2 w) c',
                            n1=int(n_sample**0.5))
    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(args.res_path, 'res.png'), imgs)
    
