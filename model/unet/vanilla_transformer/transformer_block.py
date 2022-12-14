from torch import nn

from .MHSA import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
        return self.norm_2(self.linear(y) + y)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list) # nn.ModuleList()被设计用来存储任意数量的nn. module

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
