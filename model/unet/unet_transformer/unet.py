import torch
import torch.nn as nn
from einops import rearrange

from .bottleneck_layer import Bottleneck
from .decoder import Up, SignleConv
from model.unet.vit import *

# 添加通道注意力机制，弥补自注意力机制仅仅考虑空间自适应性而忽略通道维度上的自适应性（SENet等网络中，已经证明了通道注意力的重要性）
class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out

class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=12,
                 vit_heads=4,
                 vit_dim_linear_mhsa_block=1024,
                 ):
# TransUnet(in_channels=1, img_dim=512, vit_blocks=6, vit_dim_linear_mhsa_block=512, classes=1)
        """
        Args:
            img_dim: the img dimension 图片维度:（img_dim * img_dim）
            in_channels: channels of the input 输入的通道数
            classes: desired segmentation classes 所需的分割类别
            vit_blocks: MHSA blocks of ViT
            vit_heads: number of MHSA heads
            vit_dim_linear_mhsa_block: MHSA MLP dimension
        """
        super().__init__()
        self.inplanes = 128
        vit_channels = self.inplanes * 8  # 128*8=1024

        # Not clear how they used resnet arch. since the first input after conv
        # must be 128 channels and half spat dims.
        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)

        self.img_dim_vit = img_dim // 16    #self.img_dim_vit = 512/16 = 32
        self.vit = ViT(img_dim=self.img_dim_vit,  #img_dim=32
                       in_channels=vit_channels,  # encoder channels    in_channels=1024
                       patch_dim=1,
                       dim=vit_channels,  # vit out channels for decoding  dim=1024
                       blocks=vit_blocks,  # blocks=6
                       heads=vit_heads,    # heads=4
                       dim_linear_block=vit_dim_linear_mhsa_block,   # dim_linear_block=512
                       classification=False)
        # 添加开始
        self.vit2 = ViT(img_dim=self.img_dim_vit,  #img_dim=32
                        in_channels=512,  # encoder channels    in_channels=512
                        patch_dim=1,
                        dim=512,  # vit out channels for decoding  dim=512
                        blocks=vit_blocks,  # blocks=6
                        heads=vit_heads,    # heads=4
                        dim_linear_block=vit_dim_linear_mhsa_block,   # dim_linear_block=512
                        classification=False)
        # 添加结束

        self.cab = ChannelAttentionBlock(1024)
        self.cab2 = ChannelAttentionBlock(1024)

        self.vit_conv = SignleConv(in_ch=vit_channels, out_ch=512)

        self.dec1 = Up(1024, 256)
        self.dec2 = Up(512, 128)
        self.dec3 = Up(256, 64)
        self.dec4 = Up(64, 16)
        self.conv1x1 = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        # ResNet 50-like encoder
        x2 = self.init_conv(x)  # 128,64,64(原始代码得输出维度),   1,512,512---->128,256,256(换到ROSE数据集后的输出维度)
        x4 = self.conv1(x2)     # 256,32,32(原始代码得输出维度),   128,256,256---->256,128,128(换到ROSE数据集后的输出维度)
        x8 = self.conv2(x4)     # 512,16,16(原始代码得输出维度),   256,128,128---->512,64,64(换到ROSE数据集后的输出维度)
        x16 = self.conv3(x8)    # 1024,8,8(原始代码得输出维度),    512,64,64---->1024,32,32(换到ROSE数据集后的输出维度)
        y1 = self.vit(x16)       # 1024,32,32---->(32,32),1024
        y1 = rearrange(y1, 'b (x y) dim -> b dim x y ', x=self.img_dim_vit, y=self.img_dim_vit)  # x=32,  y=32,
        # (32,32),1024---->1024,32,32
        # 添加通道注意力开始
        y2 = self.cab(x16)
        # 添加通道注意力结束

        y3 = self.vit_conv(y1 + y2)    # 1024,32,32---->512,32,32

        # 添加开始
        y4 = self.vit2(y3)
        y4 = rearrange(y4, 'b (x y) dim -> b dim x y ', x=self.img_dim_vit, y=self.img_dim_vit)
        # 添加结束

        # 添加通道注意力开始
        y5 = self.cab2(y3)
        # 添加通道注意力结束
        y = y4 + y5

        y = self.dec1(y, x8)    # 256,16,16(原始代码得输出维度)   1024,32,32---->256,64,64
        y = self.dec2(y, x4)    # 512,64,64---->128,128,128
        y = self.dec3(y, x2)    # 256,128,128---->64,256,256
        y = self.dec4(y)        # 64,256,256---->16,512,512
        return self.conv1x1(y)  # 16,512,512---->1,512,512
