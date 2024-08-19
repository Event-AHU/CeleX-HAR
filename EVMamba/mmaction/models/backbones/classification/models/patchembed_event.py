import numpy as np
import torch.nn as nn

from timm.models.layers import to_2tuple
import torch
import torch.nn.functional as F

class PatchEmbed_event(nn.Module):
    def __init__(self, in_chans=256, embed_dim=128, kernel_size=5, stride=1, flatten=True, norm_layer=False):
        super().__init__()
        self.pos_embedding = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=1, stride=1)
        self.in_chans = in_chans
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        x = x.type(torch.cuda.FloatTensor).unsqueeze(1).permute(0,1,3,2) # torch.Size([bs, 1, 131, 4096])
        xyz = self.pos_embedding(x.squeeze(dim=1)[:, :3, :]) # torch.Size([bs, 128, 4096])
        xyz = F.relu(xyz) # torch.Size([1, 128, 4096])
        x = torch.cat([xyz, x.squeeze(dim=1)[:, 3:, :]], dim=1) # torch.Size([bs, 256, 4096])
        B, N, C = x.shape       
        H = W = int(np.sqrt(N*C//self.in_chans))  # 64
        x = x.reshape(B, self.in_chans, H, W)  # torch.Size([bs, 256, 64, 64])
        x = self.proj(x)  # torch.Size([bs, 128, 60, 60])    
        x = nn.functional.interpolate(x, size=(56, 56), mode='nearest') # torch.Size([bs, 128, 56, 56]) 
        x = x.permute(0,2,3,1)  # torch.Size([bs, 56, 56, 128]) 
        
        return x