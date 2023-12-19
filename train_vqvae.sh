
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=7310 VQGAN/train.py -opt options/train_vqvae.yml --launcher pytorch 