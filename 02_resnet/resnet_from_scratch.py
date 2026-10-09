import torch
import torch.nn as nn


# ============================================================
# 1. Basic Residual Block
# ============================================================

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()


        # ----------------------------------------------------
        # Main branch
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )


        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )


        # ----------------------------------------------------
        # Shortcut
        # ----------------------------------------------------

        if (
            stride != 1
            or
            in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                # TODO 1
                # 1x1 Conv
                # in_channels -> out_channels
                # stride=stride
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                # TODO 2
                # BatchNorm
                nn.BatchNorm2d(out_channels)

            )

        else:

            self.shortcut = nn.Identity()


    def forward(self, x):

        # shortcut branch
        identity = self.shortcut(x)


        # main branch

        # TODO 3
        # Conv1
        out = self.conv1(x)

        # TODO 4
        # BN1
        out = self.bn1(out)

        # TODO 5
        # ReLU
        out = self.relu(out)


        # TODO 6
        # Conv2
        out = self.conv2(out)

        # TODO 7
        # BN2
        out = self.bn2(out)


        # TODO 8
        # residual addition
        out += identity


        # TODO 9
        # final ReLU
        out = self.relu(out)


        return out



# ============================================================
# 2. ResNet
# ============================================================

class ResNet(nn.Module):

    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10
    ):

        super().__init__()


        # 当前 feature channels
        self.in_channels = 64


        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(
            inplace=True
        )


        # ----------------------------------------------------
        # Four stages
        # ----------------------------------------------------

        self.layer1 = self._make_layer(
            block,
            out_channels=64,
            num_blocks=num_blocks[0],
            stride=1
        )


        self.layer2 = self._make_layer(
            block,

            # TODO 10
            # out_channels
            out_channels=128,

            num_blocks=num_blocks[1],

            # TODO 11
            # stride
            stride=2
        )


        self.layer3 = self._make_layer(
            block,

            # TODO 12
            out_channels=256,

            num_blocks=num_blocks[2],

            # TODO 13
            stride=2
        )


        self.layer4 = self._make_layer(
            block,

            # TODO 14
            out_channels=512,

            num_blocks=num_blocks[3],

            # TODO 15
            stride=2
        )


        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # TODO 16
        # Linear:
        # 512 -> num_classes

        self.fc = nn.Linear(
            512,
            num_classes)



    # ========================================================
    # Build one stage
    # ========================================================

    def _make_layer(
        self,
        block,
        out_channels,
        num_blocks,
        stride
    ):


        # 第一个 block 可能下采样
        # 后面的 block stride=1

        strides = (
            [stride]
            +
            [1] * (num_blocks - 1)
        )


        layers = []


        for s in strides:

            # TODO 17
            # 创建一个 block
            #
            # input:
            # self.in_channels
            #
            # output:
            # out_channels
            #
            # stride:
            # s
            layer=block(
                self.in_channels,
                out_channels,
                stride=s
            )

            layers.append(
                layer
            )


            # TODO 18
            # 更新 self.in_channels
            self.in_channels = out_channels

        return nn.Sequential(
            *layers
        )



    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):


        # Stem

        # TODO 19
        # Conv -> BN -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    


        # Four stages

        # TODO 20
        # layer1
        x = self.layer1(x)

        # TODO 21
        # layer2
        x = self.layer2(x)

        # TODO 22
        # layer3
        x = self.layer3(x)

        # TODO 23
        # layer4
        x = self.layer4(x)



        # Global Average Pooling

        # TODO 24
        x = self.avgpool(x)


        # Flatten
        #
        # (B,512,1,1)
        # ->
        # (B,512)

        # TODO 25
        x = torch.flatten(x, 1)  # Flatten from (B, 512, 1, 1) to (B, 512)


        # Classification

        # TODO 26
        x = self.fc(x)  # Linear layer for classification

        return x



# ============================================================
# 3. ResNet18
# ============================================================

def ResNet18():

    return ResNet(
        BasicBlock,

        # TODO 27
        # ResNet18:
        # [?, ?, ?, ?]
        num_blocks=[2, 2, 2, 2]

    )



# ============================================================
# 4. Test
# ============================================================

if __name__ == "__main__":

    model = ResNet18()

    x = torch.randn(
        8,
        3,
        32,
        32
    )

    y = model(x)

    print(
        "Input:",
        x.shape
    )

    print(
        "Output:",
        y.shape
    )