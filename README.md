# VQGAN
[GitHub](https://github.com/CuddleSabe/VQGAN.git) **|** [Gitee码云](https://github.com/CuddleSabe/VQGAN.git)

## Install
'''python
git clone https://github.com/CuddleSabe/VQGAN.git
cd VQGAN
python setup.py develop
'''

## Download
[Baidu](https://pan.baidu.com/s/1-LhE70tNA0YX38YUUkDi4A?pwd=fxvu) 提取码：fxvu
[Google](https://drive.google.com/drive/folders/1tdxILPP8MIBROB79RwGuA1Bv-CymRmt7?usp=share_link)

## Train
1. ''' bash
   vim options/train_vqvae.yml
   '''
2. ''' bash
   sh train_vqvae.sh
   '''
3. '''bash
   vim options/train_vqgan.yml
   '''
5. '''bash
   sh train_vqgan.sh
   '''
## Inference
'''python
python inference_vqvae.py
'''

## Idea

<img width="1123" alt="截屏2023-12-19 16 43 59" src="https://github.com/CuddleSabe/VQGAN/assets/61224076/9e526051-b3a7-4d34-b75d-10e4a4032900">

<img width="1155" alt="截屏2023-12-19 16 44 25" src="https://github.com/CuddleSabe/VQGAN/assets/61224076/f8a137a4-9400-4cba-9c71-d4fdd169c866">

<img width="1321" alt="截屏2023-12-19 16 44 52" src="https://github.com/CuddleSabe/VQGAN/assets/61224076/4a5206ee-9783-4e65-914b-599c5c926f5a">
