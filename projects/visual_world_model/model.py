import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):

    def __init__(
        self,
        latent_dim=256
    ):
        super().__init__()

        self.conv = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(
                3, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 64 -> 32
            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 32 -> 16
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 16 -> 8
            nn.Conv2d(
                128, 256,
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


    def forward(self, x):

        x = self.conv(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        z = self.fc(x)

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

            # 4 -> 8
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 8 -> 16
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 16 -> 32
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 32 -> 64
            nn.ConvTranspose2d(
                32, 3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )


    def forward(self, z):

        x = self.fc(z)

        x = x.view(
            z.size(0),
            256,
            8,
            8
        )

        x = self.deconv(x)

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

        self.target_head = nn.Sequential(
        nn.Linear(
            latent_dim,
            128
        ),
        nn.ReLU(),

        nn.Linear(
            128,
            2
        ),

        nn.Sigmoid()
    )


    def forward(self, x):

        z = self.encoder(x)

        reconstruction = self.decoder(z)

        target_xy = self.target_head(z)

        return reconstruction, z, target_xy



class TargetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # RGB + coordinate-x + coordinate-y
        self.conv = nn.Sequential(

            # 128 -> 128
            nn.Conv2d(
                5, 32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            # 128 -> 64
            nn.Conv2d(
                32, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 64 -> 64
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            # 64 -> 32
            nn.Conv2d(
                64, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 32 -> 16
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                256
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                2
            )
        )


    def forward(self, x):

        B, C, H, W = x.shape

        # ----------------------------------------
        # coordinate maps
        # ----------------------------------------

        y_coord = torch.linspace(
            -1,
            1,
            H,
            device=x.device,
            dtype=x.dtype
        )

        x_coord = torch.linspace(
            -1,
            1,
            W,
            device=x.device,
            dtype=x.dtype
        )

        yy, xx = torch.meshgrid(
            y_coord,
            x_coord,
            indexing="ij"
        )

        coords = torch.stack(
            [xx, yy],
            dim=0
        )

        coords = coords.unsqueeze(0).expand(
            B,
            -1,
            -1,
            -1
        )

        # RGB -> RGBXY
        x = torch.cat(
            [x, coords],
            dim=1
        )

        h = self.conv(x)

        target_xy = self.head(h)

        return target_xy



# ============================================================
# Action-conditioned Latent Dynamics
#
# z_(t+1) = z_t + Delta(z_t, a_t)
# ============================================================

class LatentDynamics(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        action_dim=2,
        hidden_dim=512
    ):
        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                latent_dim + action_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                latent_dim
            )
        )


        # ----------------------------------------
        # Start approximately from:
        #
        # z_next_pred = z_t
        #
        # i.e. identity dynamics.
        #
        # Then learn only the change Delta z.
        # ----------------------------------------

        last_layer = self.net[-1]

        nn.init.zeros_(
            last_layer.weight
        )

        nn.init.zeros_(
            last_layer.bias
        )


    def forward(
        self,
        z,
        action
    ):

        # z:
        # (B,256)
        #
        # action:
        # (B,2)

        x = torch.cat(
            [
                z,
                action
            ],
            dim=1
        )


        delta_z = self.net(
            x
        )


        z_next_pred = (
            z
            +
            delta_z
        )


        return z_next_pred


# ============================================================
# No-action Latent Dynamics
#
# z_(t+1) = z_t + Delta(z_t)
#
# Used as a baseline to test whether action_t
# really contributes useful predictive information.
# ============================================================

class NoActionLatentDynamics(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512
    ):
        super().__init__()


        self.net = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                latent_dim
            )
        )


        # Start from identity baseline:
        #
        # z_next_pred = z_t

        last_layer = self.net[-1]

        nn.init.zeros_(
            last_layer.weight
        )

        nn.init.zeros_(
            last_layer.bias
        )


    def forward(
        self,
        z
    ):

        delta_z = self.net(
            z
        )

        z_next_pred = (
            z
            +
            delta_z
        )

        return z_next_pred


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

    recon, z = model(x)

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