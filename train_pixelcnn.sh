
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=7310 VQGAN/train.py -opt options/train_pixelcnn.yml --launcher pytorch 