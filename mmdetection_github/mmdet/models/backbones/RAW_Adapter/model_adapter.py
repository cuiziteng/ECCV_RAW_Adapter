import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmengine.model import BaseModule
from typing import Any, Callable, List, Optional, Type, Union

def conv7x7(in_planes: int, out_planes: int, stride: int = 3, groups: int = 1,  padding: int = 3, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# The 1-st Model-level Adapter Generation 
class Model_level_Adapeter(BaseModule):
    def __init__(self, in_c=3, in_dim=24, w_lut=True):
        super(Model_level_Adapeter, self).__init__()
        self.conv_1 = conv3x3(in_c, in_c, 2)
        self.conv_2 = conv3x3(in_c, in_c, 2)
        self.conv_3 = conv3x3(in_c, in_c, 2)
        self.w_lut = w_lut
        if self.w_lut:  # With implicit 3DLUT: I1, I2, I3, I4
            self.conv_4 = conv3x3(in_c, in_c, 2)
            self.uni_conv = conv7x7(4*in_c, in_dim, 2, padding=3)
        else:  # Without implicit 3DLUT: I1, I2, I3
            self.uni_conv = conv7x7(3*in_c, in_dim, 2, padding=3)

        self.res_1 = BasicBlock(inplanes=in_dim, planes=in_dim)
        self.res_2 = BasicBlock(inplanes=in_dim, planes=in_dim)

    def forward(self, IMGS):
        if self.w_lut:
            adapter = torch.cat([self.conv_1(IMGS[0]), self.conv_2(IMGS[1]), self.conv_3(IMGS[2]), self.conv_4(IMGS[3])], dim=1)

        else:
            adapter = torch.cat([self.conv_1(IMGS[0]), self.conv_2(IMGS[1]), self.conv_3(IMGS[2])], dim=1)
        
        adapter = self.uni_conv(adapter)
        adapter = self.res_1(adapter)   # Residual Block 1 
        adapter = self.res_2(adapter)   # Residual Block 2
        return adapter

# Feature Merge Block
class Merge_block(BaseModule):

    def __init__(self, fea_c, ada_c, mid_c, return_ada=True):
        super(Merge_block, self).__init__()

        self.conv_1 = conv1x1(fea_c+ada_c, mid_c, 1)
        self.conv_2 = conv1x1(mid_c, fea_c, 1)
        self.return_ada = return_ada
        if self.return_ada:
            self.conv_3 = conv3x3(mid_c, ada_c*2, stride=2)
        
    
    def forward(self, fea, adapter, ratio=1.0):

        res = fea
        fea = torch.cat([fea, adapter], dim=1)
        
        fea = self.conv_1(fea)
        ada = self.conv_2(fea)
        fea_out = ratio*ada + res
        
        if self.return_ada: # return adapter for next level
            ada = self.conv_3(fea)
            return fea_out, ada
        
        else:
            return fea_out, None

if __name__ == "__main__":
    
    model_adapter = Model_level_Adapeter(in_c=3, w_lut=True)
    print(sum(param.numel() for param in model_adapter.parameters()))
    IMGS = [torch.randn(2,3,416,608), torch.randn(2,3,416,608), torch.randn(2,3,416,608), torch.randn(2,3,416,608)]
    adapter = model_adapter(IMGS)

    fea_1 = torch.randn(2, 256, 104, 152)
    fea_2 = torch.randn(2 ,512, 52, 76)
    fea_3 = torch.randn(2, 1024, 26, 38)

    merge_1 = Merge_block(fea_c=256, ada_c=24, mid_c=64, return_ada=True)
    merge_2 = Merge_block(fea_c=512, ada_c=48, mid_c=64, return_ada=True)
    merge_3 = Merge_block(fea_c=1024, ada_c=96, mid_c=128, return_ada=False)
    
    print(sum(param.numel() for param in  merge_1.parameters())+
         sum(param.numel() for param in  merge_2.parameters())+
         sum(param.numel() for param in  merge_3.parameters()))
    
    fea_out_1, adapter = merge_1(fea_1, adapter)
    
    fea_out_2, adapter = merge_2(fea_2, adapter)
    
    fea_out_3, adapter = merge_3(fea_3, adapter)