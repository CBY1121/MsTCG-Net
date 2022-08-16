import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from model.unet import TransformerEncoder
from model.unet.common import expand_to_batch


class SPP_Q(nn.Module):
    def __init__(self,in_ch,out_ch,down_scale,ks=3):
        super(SPP_Q, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=1, padding=ks // 2,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.Down = nn.Upsample(scale_factor=down_scale,mode="bilinear")  # scale_factor：指定输出为输入的多少倍数

    def forward(self, x):
        x_d = self.Down(x)
        x_out = self.Conv(x_d)
        return x_out

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


class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0, transformer=None, classification=True):
        """
        Minimal re-implementation of ViT
        Args:
            img_dim: the spatial image size空间图像大小
            in_channels: number of img channels图片的通道数
            patch_dim: desired patch dim  patch的维度
            num_classes: classification task classes 分几类
            dim: the linear layer's dim to project the patches for MHSA   MHSA中，线性层投影patch的维度
            blocks: number of transformer blocks  transformer块的数量
            heads: number of heads    heads的数量
            dim_linear_block: inner dim（内部维度） of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation 如果您想提供另一个transfomer实现
            classification: creates an extra CLS token that we will index in the final classification layer
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        self.p = patch_dim
        self.classification = classification
        # tokens = number of patches
        tokens = (img_dim // patch_dim) ** 2  # img_dim=32, patch_dim=1
        self.token_dim = in_channels * (patch_dim ** 2)  # 1024
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings   #nn.Linear()设置网络中的全连接层
        self.project_patches = nn.Linear(self.token_dim, dim)  # 1024,1024
        # 将四维张量转换为二维张量之后，才能作为全连接层的输入.
        # nn.Linear()是用于设置网络中的全连接层的，需要注意在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，
        # 形状通常为[batch_size, size]，不同于卷积层要求输入输出是四维张量。

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的，
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
        self.mlp_head = nn.Linear(dim, num_classes)

        if transformer is None:
            self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        # einops.rearrange 是一个便于阅读的多维张量智能元素重新排序。该操作包括转置（轴排列）、重塑（视图）、挤压、解压、堆叠、连接等操作的功能。
        # patch_x = p = 1, patch_y = p = 1, x = h/p, y = w/p

        batch_size, tokens, _ = img_patches.shape   # batch_size, c=1024, 1*32, 1*32 ----> batch_size, tokens=32*32, 1024

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim] = [batch, 32*32, 1024]
        y = self.transformer(patch_embeddings, mask)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 1:, :]   # self.classification=False

class SignleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same双卷积块，保持空间大小相同
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(SignleConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Encoder_Pos(nn.Module):
    def __init__(self, n_dims, width=32, height=32, filters=[32,64,128,256]):  # n_dims = 512
        super(Encoder_Pos, self).__init__()
        print("================= Multi_Head_Encoder_Decoder =================")

        self.chanel_in = n_dims  # 512
        self.Dim_head = 256  # dim_head: dim head in case you want to define it. defaults to dim/heads
        self.rel_h = nn.Parameter(torch.randn([1, n_dims//8, height, 1]), requires_grad=True)  # 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//8, 1, width]), requires_grad=True)

        self.SPP_Q_0 = SPP_Q(in_ch=filters[0],out_ch=n_dims,down_scale=1/16,ks=3)
        self.SPP_Q_1 = SPP_Q(in_ch=filters[1],out_ch=n_dims,down_scale=1/8,ks=3)
        self.SPP_Q_2 = SPP_Q(in_ch=filters[2],out_ch=n_dims,down_scale=1/4,ks=3)
        self.SPP_Q_3 = SPP_Q(in_ch=filters[3],out_ch=n_dims,down_scale=1/2,ks=3)


        self.query_conv = nn.Conv2d(in_channels = n_dims , out_channels = n_dims//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = n_dims , out_channels = n_dims//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = n_dims , out_channels = n_dims , kernel_size= 1)

        self.scale_factor = (np.log2(self.Dim_head) ** -1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.vit = ViT(img_dim=32,  # img_dim=self.img_dim_vit=32
                       in_channels=512,  # encoder channels    in_channels=vit_channels=1024
                       patch_dim=1,
                       dim=512,  # vit out channels for decoding  dim=vit_channels=1024
                       blocks=6,  # blocks=vit_blocks=6
                       heads=4,  # heads=vit_heads=4
                       dim_linear_block=512,  # dim_linear_block=vit_dim_linear_mhsa_block=512
                       classification=False)
        self.cab = ChannelAttentionBlock(512)
        self.vit_conv = SignleConv(in_ch=512, out_ch=512)

    def forward(self, x,x_list):
        m_batchsize, C, width, height = x.size()
        Multi_X = self.SPP_Q_0(x_list[0]) + self.SPP_Q_1(x_list[1]) + self.SPP_Q_2(x_list[2]) + self.SPP_Q_3(x_list[3])  # 得到FA
        proj_query = self.query_conv(Multi_X).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # 得到K

        energy_content = torch.bmm(proj_query, proj_key)  # torch.bmm计算两个tensor的矩阵乘法得到E
        # 添加一个dk
        energy_content = energy_content * self.scale_factor

        content_position = (self.rel_h + self.rel_w).view(1, self.chanel_in//8, -1) # 得到PE
        content_position = torch.matmul(proj_query,content_position)  # 得到EP
        energy = energy_content + content_position
        attention = self.softmax(energy)  # 得到Att
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height) # 得到FM

        y1 = self.vit(x)  # 1024,32,32---->(32,32),1024
        y1 = rearrange(y1, 'b (x y) dim -> b dim x y ', x=32, y=32)  # x=self.img_dim_vit=32,  y=self.img_dim_vit=32,
        # (32,32),1024---->1024,32,32
        y2 = self.cab(x)
        y3 = self.vit_conv(y1 + y2)  # 1024,32,32---->512,32,32

        # out = self.gamma * out + x  # 得到FEn
        out = self.gamma * out
        out = out + y3 + x


        return out, attention


class Decoder_Pos(nn.Module):
    def __init__(self, n_dims, width=32, height=32):
        super(Decoder_Pos, self).__init__()
        print("================= Multi_Head_Decoder =================")

        self.chanel_in = n_dims
        self.rel_h = nn.Parameter(torch.randn([1, n_dims//8, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//8, 1, width]), requires_grad=True)
        self.query_conv = nn.Conv2d(in_channels=n_dims, out_channels=n_dims // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=n_dims, out_channels=n_dims // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=n_dims, out_channels=n_dims, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,x_encoder):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x_encoder).view(m_batchsize, -1, width * height)

        energy_content = torch.bmm(proj_query, proj_key) # 得到ED


        content_position = (self.rel_h + self.rel_w).view(1, self.chanel_in//8, -1)
        content_position = torch.matmul(proj_query,content_position) # 得到EPD

        energy = energy_content+content_position
        attention = self.softmax(energy) # 得到Att
        proj_value = self.value_conv(x_encoder).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height) # 得到FMD

        out = self.gamma * out + x
        return out, attention





class MsTNL(nn.Module):
    def __init__(self,train_dim,filters=[32,64,128,256]):
        print("============= MsTNL_TC =============")
        super(MsTNL, self).__init__()
        self.encoder = Encoder_Pos(train_dim,width=32,height=32,filters=filters)  # train_dim=512
        # self.decoder = Decoder_Pos(train_dim,width=32,height=32)

    def forward(self, x, x_list):

        x_encoder,att_en = self.encoder(x, x_list)
        # x_out,att_de = self.decoder(x,x_encoder)

        # return x_out
        return x_encoder
