import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class VisualEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # TODO 1
        # 去掉最后的 classification layer
        #
        # 提示：
        # ResNet 的 fc 可以替换成某个
        # “什么都不做”的 layer
        model.fc = nn.Identity()

        self.encoder = model


    def forward(self, x):

        # TODO 2
        # 得到 image embedding

        z = self.encoder(x)

        return z


encoder = VisualEncoder()

x = torch.randn(
    8,
    3,
    224,
    224
)

z = encoder(x)

print("Input:", x.shape)
print("Embedding:", z.shape)