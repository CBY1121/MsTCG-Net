import torch
import torch.nn as nn
from einops import rearrange

from model.unet import TransformerEncoder
from ..common import expand_to_batch

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
