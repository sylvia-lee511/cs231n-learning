import torch
import torch.nn as nn


class VisualEncoder(nn.Module):

    def __init__(
        self,
        latent_dim=256
    ):
        super().__init__()


        self.conv = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(
                3,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 64 -> 32
            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 32 -> 16
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 16 -> 8
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )


        self.fc = nn.Linear(
            256 * 8 * 8,
            latent_dim
        )


    def forward(
        self,
        x
    ):

        x = self.conv(
            x
        )


        x = torch.flatten(
            x,
            start_dim=1
        )


        z = self.fc(
            x
        )


        return z


class VisualDecoder(nn.Module):

    def __init__(
        self,
        latent_dim=256
    ):
        super().__init__()


        self.fc = nn.Linear(
            latent_dim,
            256 * 8 * 8
        )


        self.deconv = nn.Sequential(

            # 8 -> 16
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 16 -> 32
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 32 -> 64
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 64 -> 128
            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        z
    ):

        x = self.fc(
            z
        )


        x = x.view(
            z.size(0),
            256,
            8,
            8
        )


        x = self.deconv(
            x
        )


        return x


class VisualAutoencoder(nn.Module):

    def __init__(
        self,
        latent_dim=256
    ):
        super().__init__()


        self.encoder = VisualEncoder(
            latent_dim
        )


        self.decoder = VisualDecoder(
            latent_dim
        )


        # ----------------------------------------------------
        # Auxiliary target-position head
        #
        # IMPORTANT:
        # No Sigmoid here.
        #
        # The label is the real Reacher target coordinate
        # from the simulator (obs[4:6]), which can be negative.
        #
        # This head is used only during training/evaluation.
        # The RGB image remains the only encoder input.
        # ----------------------------------------------------

        self.target_head = nn.Sequential(

            nn.Linear(
                latent_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                2
            )
        )


    def forward(
        self,
        x
    ):

        z = self.encoder(
            x
        )


        reconstruction = self.decoder(
            z
        )


        target_xy_world = (
            self.target_head(
                z
            )
        )


        return (
            reconstruction,
            z,
            target_xy_world
        )


if __name__ == "__main__":

    model = VisualAutoencoder(
        latent_dim=256
    )


    x = torch.randn(
        8,
        3,
        128,
        128
    )


    (
        recon,
        z,
        target_xy_world
    ) = model(
        x
    )


    print(
        "Input:",
        x.shape
    )

    print(
        "Latent:",
        z.shape
    )

    print(
        "Recon:",
        recon.shape
    )

    print(
        "Target XY:",
        target_xy_world.shape
    )
