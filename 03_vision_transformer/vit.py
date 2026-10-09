import torch
import torch.nn as nn
from transformer import TransformerBlock

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size

        # 每边 patch 数量
        self.grid_size = (
            image_size // patch_size
        )

        # 总 patch 数量
        self.num_patches = (
            self.grid_size ** 2
        )


        # ==========================================
        # Patchify + Projection
        # ==========================================

        # TODO 1
        # Conv2d:
        #
        # in_channels -> embed_dim
        # kernel_size = patch_size
        # stride = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self, x):
        """
        x:
        (B,C,H,W)
        """


        # ==========================================
        # 1. Patch Projection
        # ==========================================

        # TODO 2
        #
        # (B,C,H,W)
        # ->
        # (B,D,H/P,W/P)

        x = self.proj(x)


        # ==========================================
        # 2. Flatten spatial dimension
        # ==========================================

        # TODO 3
        #
        # (B,D,H/P,W/P)
        # ->
        # (B,D,N)

        x = nn.Flatten(
            start_dim=2,
            end_dim=3
        )(x)


        # ==========================================
        # 3. Move embedding dimension last
        # ==========================================

        # TODO 4
        #
        # (B,D,N)
        # ->
        # (B,N,D)

        x = x.transpose(1,2)


        return x



class ViTEmbedding(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = (
            self.patch_embed.num_patches
        )


        # ==========================================
        # CLS token
        # ==========================================

        # TODO 1
        #
        # 可学习参数
        # shape:
        # (1,1,D)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )


        # ==========================================
        # Position embedding
        # ==========================================

        # TODO 2
        #
        # shape:
        # (1,N+1,D)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )


    def forward(self, x):

        # ==========================================
        # Patch embedding
        # ==========================================

        # TODO 3
        #
        # (B,3,H,W)
        # ->
        # (B,N,D)

        x = self.patch_embed(x)


        B = x.shape[0]


        # ==========================================
        # Expand CLS token
        # ==========================================

        # TODO 4
        #
        # (1,1,D)
        # ->
        # (B,1,D)

        cls_tokens = self.cls_token.expand(B, -1, -1)


        # ==========================================
        # Concatenate
        # ==========================================

        # TODO 5
        #
        # (B,1,D)
        # +
        # (B,N,D)
        #
        # ->
        # (B,N+1,D)

        x = torch.cat([cls_tokens, x], dim=1)


        # ==========================================
        # Add position embedding
        # ==========================================

        # TODO 6

        x = x + self.pos_embed

        return x


class VisionTransformer(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()


        # ==========================================
        # 1. Image -> Token Sequence
        # ==========================================

        # TODO 1
        #
        # 使用你已经写好的 ViTEmbedding

        self.embedding = ViTEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )


        # ==========================================
        # 2. Transformer Blocks
        # ==========================================

        # TODO 2
        #
        # depth 个 TransformerBlock
        #
        # 使用 nn.ModuleList

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)])


        # ==========================================
        # 3. Final LayerNorm
        # ==========================================

        # TODO 3

        self.norm = nn.LayerNorm(embed_dim)


        # ==========================================
        # 4. Classification Head
        # ==========================================

        # TODO 4
        #
        # embed_dim -> num_classes

        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):


        # ==========================================
        # Image -> tokens
        # ==========================================

        # TODO 5

        x = self.embedding(x)


        # ==========================================
        # Transformer Encoder
        # ==========================================

        # TODO 6
        #
        # 依次经过所有 blocks

        for block in self.blocks:
            x= block(x)


        # ==========================================
        # Final Norm
        # ==========================================

        # TODO 7
        x = self.norm(x)


        # ==========================================
        # CLS token
        # ==========================================

        # TODO 8
        #
        # x:
        # (B,N+1,D)
        #
        # cls:
        # (B,D)

        cls = x[:, 0]


        # ==========================================
        # Classification
        # ==========================================

        # TODO 9

        logits = self.head(cls)


        return logits


if __name__ == "__main__":

    x = torch.randn(
        8,
        3,
        32,
        32
    )

    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,

        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4
    )

    logits = model(x)

    print("Input :", x.shape)
    print("Output:", logits.shape)
