import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


@ARCH_REGISTRY.register()
class VQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
            nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), 
            ResidualBlock(dim)
        )
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2
        
    def encode(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, 4)
        return self.encoder(x)
    
    def decode(self, ze):
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        decoder_input = ze + (zq - ze).detach()
        # decoder_input = ze + (zq - ze).detach()
        x_hat = self.decoder(decoder_input)
        x_hat = torch.nn.functional.pixel_shuffle(x_hat, 4)
        return x_hat, ze, zq
    
    
    def encode_idx(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, 4)
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor
    
    
    def decode_idx(self, nearest_neighbor):
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        decoder_input = zq
        x_hat = self.decoder(decoder_input)
        x_hat = torch.nn.functional.pixel_shuffle(x_hat, 4)
        return x_hat
    
        
    def forward(self, x):
        ze = self.encode(x)
        return self.decode(ze)
        
@ARCH_REGISTRY.register()
class VQVAE_multi_codebook(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, n_codebook):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
            nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), 
            ResidualBlock(dim)
        )
        self.vq_embedding = nn.Embedding(n_embedding, dim//n_codebook)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2
        self.n_codebook = n_codebook
        
    def encode(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, 4)
        return self.encoder(x)
    
    def decode(self, ze):
        embedding = self.vq_embedding.weight.data
        ze = torch.cat(ze.chunk(self.n_codebook, dim=1), dim=0)
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        decoder_input = ze + (zq - ze).detach()
        decoder_input = torch.cat(decoder_input.chunk(self.n_codebook, dim=0), dim=1)
        x_hat = self.decoder(decoder_input)
        x_hat = torch.nn.functional.pixel_shuffle(x_hat, 4)
        return x_hat, ze, zq
    
    def encode_idx(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, 4)
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data
        ze = torch.cat(ze.chunk(self.n_codebook, dim=1), dim=0)
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        nearest_neighbor = torch.cat(nearest_neighbor.chunk(self.n_codebook, dim=0), dim=1)
        return nearest_neighbor
    
    
    def decode_idx(self, nearest_neighbor):
        nearest_neighbor = torch.cat(nearest_neighbor.chunk(self.n_codebook, dim=1), dim=0)
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        decoder_input = zq
        decoder_input = torch.cat(decoder_input.chunk(self.n_codebook, dim=0), dim=1)
        x_hat = self.decoder(decoder_input)
        x_hat = torch.nn.functional.pixel_shuffle(x_hat, 4)
        return x_hat
    
    def decode_zq(self, zq):
        decoder_input = zq
        decoder_input = torch.cat(decoder_input.chunk(self.n_codebook, dim=0), dim=1)
        x_hat = self.decoder(decoder_input)
        x_hat = torch.nn.functional.pixel_shuffle(x_hat, 4)
        return x_hat

        
    def forward(self, x):
        ze = self.encode(x)
        return self.decode(ze)



if __name__ == '__main__':
    import time
    from tqdm import tqdm
    vqvae = VQVAE_multi_codebook(input_dim=48, dim=64, n_embedding=256, n_codebook=64)
    vqvae.to('cuda')
    vqvae.eval()
    all_time = 0
    iter_time = 100
    patch_size=64
    with torch.no_grad():
        for i in range(5):
            # img = torch.rand(1, 3, int(1920/4+1)*4, int(1080/4+1)*4).to('cuda')
            img = torch.rand(16, 3, 256, 256).to('cuda')
            sr = vqvae.encode_idx(img)
            print(sr.shape)
        
        for i in tqdm(range(iter_time)):
            # img = torch.rand(1, 3, int(1920/4+1)*4, int(1080/4+1)*4).to('cuda')
            img = torch.rand(1, 3, 1024, 1024).to('cuda')
            torch.cuda.synchronize()
            t0 = time.time()
            sr = vqvae(img)
            torch.cuda.synchronize()
            t1 = time.time()
            all_time += (t1-t0)*1000
        all_time /= iter_time
        
    print(all_time)