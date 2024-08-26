import imp
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from model.blocks import Mlp
from einops.layers.torch import Rearrange
import math

# Predictor P_K
class Kernel_Predictor(nn.Module):
    def __init__(self, dim, mode='low', num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # Query Adaptive Learning (QAL)
        self.q = nn.Parameter(torch.rand((1, 4, dim)), requires_grad=True)

        self.kv_downsample = nn.Sequential(
            nn.Conv2d(3, dim // 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim),
        )
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.down = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Basic Parameters Number
        if mode == 'low':
            self.gain_base = nn.Parameter(torch.FloatTensor([3]), requires_grad=True)
        else:
            self.gain_base = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        

        self.r1_base = nn.Parameter(torch.FloatTensor([3]), requires_grad=False)
        self.r2_base = nn.Parameter(torch.FloatTensor([2]), requires_grad=False)

    def forward(self, x):
        d_x = self.kv_downsample(x).flatten(2).transpose(1, 2)
        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, 4, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.down(out).squeeze(-1)
        
        out = torch.unbind(out, 1)
        r1, r2, gain, sigma = out[0], out[1], out[2], out[3]
        r1 = 0.1 * r1 +  self.r1_base
        r2 = 0.1 * r2 +  self.r2_base
        
        gain =gain + self.gain_base
        
        return r1, r2, gain, self.sigmoid(sigma)
        

# Predictor P_M
class Matrix_Predictor(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # Query Adaptive Learning (QAL)
        self.q = nn.Parameter(torch.rand((1, 9 + 1, dim)), requires_grad=True)
        self.kv_downsample = nn.Sequential(
            nn.Conv2d(3, dim // 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(dim),
        )
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.down = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.ccm_base = nn.Parameter(torch.eye(3), requires_grad=False)

    def forward(self, x):
        d_x = self.kv_downsample(x).flatten(2).transpose(1, 2)
        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, 9 + 1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.down(out)
        out, distance = out[:, :9, :], out[:, 9:, :].squeeze(-1)
        out = out.view(B, 3, 3)
        # print(self.ccm_base)
        # print(out)
        
        ccm_matrix = 0.1 * out + self.ccm_base
        distance = self.relu(distance) + 1

        return ccm_matrix, distance

## AAAI 2024 NILUT, we change the channel number to avoid much FLOPs
class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
    def __init__(self, in_features=3, hidden_features=32, hidden_layers=3, out_features=3, res=True):
        super().__init__()
        
        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity):
        output = self.net(intensity)
        if self.res:
            output = output + intensity
            output = torch.clamp(output, 0.,1.)
        return output


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='4'
    #net = Local_pred_new().cuda()
    img = torch.Tensor(1, 400, 600, 3)
    # global_net = single_attention(dim=32)
    # out, gamma = global_net(img)
    # flops = FlopCountAnalysis(global_net, img)
    # print("GFLOPs: ", flops.total()/1e9)
    # print('total parameters:', sum(param.numel() for param in global_net.parameters()))
    # print(gamma.shape, out.shape)
    net = NILUT(hidden_features=128)
    print('LUT parameters:', sum(param.numel() for param in net.parameters()))
    flops = FlopCountAnalysis(net, img)
    print("GFLOPs: ", flops.total()/1e9)
    # P_M = Matrix_Predictor(dim=64)
    # distance = P_M(img)
    #print('111', ccm_matrix)
    # print('222', distance.shape)

    # print(ccm_matrix.shape, distance.shape)
    # print('P_M parameters:', sum(param.numel() for param in P_M.parameters()))
    

    # P_K = Kernel_Predictor(dim=64)
    # r1, r2, gain, sigma = P_K(img)
    
    

    # print(r1)
    # print(r2)
    # flops = FlopCountAnalysis(P_K, img)
    # print("GFLOPs: ", flops.total()/1e9)

    # print(paras.shape, range.shape)
    # print('P_K parameters:', sum(param.numel() for param in P_K.parameters()))
    '''
    img_reshape = img.permute(0,2,3,1) # (B, C, H, W) --> (B, H, W, C)
    
    print(img.permute(0,2,3,1).shape)
    print(img_reshape.shape)

    LUT = NILUT()
    out_image = LUT(img_reshape).permute(0,3,1,2)
    
    print('LUT parameters:', sum(param.numel() for param in LUT.parameters()))
    print(out_image.shape)
    flops = FlopCountAnalysis(LUT, img_reshape)
    print("GFLOPs: ", flops.total()/1e9)
    '''


    # print(output.shape)
    