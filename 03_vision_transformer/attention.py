import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = (
            embed_dim // num_heads
        )


        # ======================================
        # Q K V projection
        # ======================================

        self.W_q = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False
        )

        self.W_k = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False
        )

        self.W_v = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False
        )


        # output projection
        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )


    def forward(self, x):
        """
        x:
        (B, N, D)
        """

        B, N, D = x.shape


        # ======================================
        # 1. Q K V
        # ======================================

        # TODO 1
        q = self.W_q(x)

        # TODO 2
        k = self.W_k(x)

        # TODO 3
        v = self.W_v(x)


        # 当前：
        #
        # q,k,v:
        # (B,N,D)


        # ======================================
        # 2. Split heads
        # ======================================

        # (B,N,D)
        # ->
        # (B,N,h,d_h)

        # TODO 4
        q = torch.reshape(q,(B,N,self.num_heads,self.head_dim))

        # TODO 5
        k = torch.reshape(k,(B,N,self.num_heads,self.head_dim))

        # TODO 6
        v = torch.reshape(v,(B,N,self.num_heads,self.head_dim))


        # ======================================
        # 3. Move heads forward
        # ======================================

        # (B,N,h,d_h)
        # ->
        # (B,h,N,d_h)

        # TODO 7
        q = q.transpose(1, 2)  # (B, h, N, d_h)

        # TODO 8
        k = k.transpose(1, 2)

        # TODO 9
        v = v.transpose(1, 2)


        # ======================================
        # 4. Attention scores
        # ======================================

        # Q @ K^T
        #
        # (B,h,N,d)
        # @
        # (B,h,d,N)
        #
        # ->
        # (B,h,N,N)

        # TODO 10
        scores = q@k.transpose(-2, -1) 


        # scale
        # TODO 11
        scores = scores / math.sqrt(self.head_dim)

        # ======================================
        # 5. Softmax
        # ======================================

        # TODO 12
        attention = torch.softmax(scores, dim=-1)


        # ======================================
        # 6. Weighted sum
        # ======================================

        # (B,h,N,N)
        # @
        # (B,h,N,d)
        #
        # ->
        # (B,h,N,d)

        # TODO 13
        out = attention @ v


        # ======================================
        # 7. Merge heads
        # ======================================

        # (B,h,N,d)
        # ->
        # (B,N,h,d)

        # TODO 14
        out = out.transpose(1, 2)


        # (B,N,h,d)
        # ->
        # (B,N,D)

        # TODO 15
        out = torch.reshape(out, (B, N, self.embed_dim))


        # ======================================
        # 8. Output projection
        # ======================================

        # TODO 16
        out = self.out_proj(out)


        return out, attention

B = 2
N = 4
D = 8
H = 2

x = torch.randn(
    B,
    N,
    D
)

mha = MultiHeadSelfAttention(
    embed_dim=D,
    num_heads=H
)

out, weights = mha(x)

print("Input:")
print(x.shape)

print("\nOutput:")
print(out.shape)

print("\nAttention:")
print(weights.shape)

print("\nHead 0 attention:")
print(weights[0, 0])

print("\nRow sums:")
print(
    weights[0, 0].sum(dim=-1)
)