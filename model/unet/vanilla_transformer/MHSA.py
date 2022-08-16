import numpy as np
import torch
from einops import rearrange
from torch import nn

def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # softmax第一步：将模型的预测结果转化到指数函数上，这样保证了概率的非负性。
    # 为了确保各个预测结果的概率之和等于1。我们只需要将转换后的结果进行归一化处理。
    # 方法就是将转化后的结果除以所有转化后结果之和，可以理解为转化后结果占总数的百分比。这样就得到近似的概率。

    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads  # 256*4=1024
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)  # 1024,1024*3
        self.W_0 = nn.Linear(_dim, dim, bias=False)  # 1024,1024
        self.scale_factor = (np.log(self.dim_head) ** -1)
        # self.scale_factor = (self.dim_head ** -0.5) * 1/2  # 修改之前：self.scale_factor = self.dim_head ** -0.5
        # self.scale_factor = (self.dim_head ** -2)


    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ] = [batch, 32*32, 1024*3*4 ]

        # decomposition to q,v,k and cast to tuple 分解为 q,v,k 并转换为元组
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        # 转换为元组之前的结果形状将是：[3,batch,heads,tokens,dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))  # 重新排列形状,h=4

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
