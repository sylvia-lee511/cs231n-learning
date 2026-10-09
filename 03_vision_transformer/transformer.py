from attention import MultiHeadSelfAttention
import torch
import torch.nn as nn



class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()

        # ==========================================
        # LayerNorm 1
        # ==========================================

        self.norm1 = nn.LayerNorm(
            embed_dim
        )


        # ==========================================
        # Multi-Head Attention
        # ==========================================

        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )


        # ==========================================
        # LayerNorm 2
        # ==========================================

        self.norm2 = nn.LayerNorm(
            embed_dim
        )


        # ==========================================
        # MLP
        # ==========================================

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        self.mlp = nn.Sequential(

            # TODO 1
            # Linear:
            # embed_dim -> hidden_dim
            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            # TODO 2
            # GELU
            nn.GELU(),

            # TODO 3
            # Dropout
            nn.Dropout(dropout),

            # TODO 4
            # Linear:
            # hidden_dim -> embed_dim
            nn.Linear(
                hidden_dim,
                embed_dim
            ),

            # TODO 5
            # Dropout
            nn.Dropout(dropout)
        )

        


    def forward(self, x):
        """
        x:
        (B, N, D)
        """


        # ==========================================
        # Attention branch
        # ==========================================

        # TODO 6
        # LayerNorm

        norm_x = self.norm1(x)


        # TODO 7
        # Multi-Head Attention
        #
        # 注意你的 attention
        # return:
        # out, attention

        attn_out = self.attn(norm_x)[0]


        # TODO 8
        # Residual
        #
        # x = x + ...
        x = x + attn_out

        # ==========================================
        # MLP branch
        # ==========================================

        # TODO 9
        # LayerNorm

        norm_x = self.norm2(x)


        # TODO 10
        # MLP

        mlp_out = self.mlp(norm_x)


        # TODO 11
        # Residual
        x = x + mlp_out

        return x


