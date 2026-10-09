from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import (
    Dataset,
    DataLoader
)

from dataset_modified import (
    preprocess_reacher_frame
)

from model import (
    VisualAutoencoder,
    LatentDynamics,
    NoActionLatentDynamics
)


# ============================================================
# 1. Config
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent

REPO_ROOT = (
    SCRIPT_DIR.parent
)


DATA_PATH = (
    REPO_ROOT
    / "data_visual_world_model"
    / "reacher_visual_random.npz"
)


AE_CHECKPOINT = (
    SCRIPT_DIR
    / "best_visual_autoencoder_worldtarget.pt"
)


ACTION_CHECKPOINT = (
    SCRIPT_DIR
    / "best_latent_dynamics.pt"
)


NOACTION_CHECKPOINT = (
    SCRIPT_DIR
    / "best_noaction_dynamics.pt"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)


LATENT_DIM = 256
ACTION_DIM = 2
HIDDEN_DIM = 512


HORIZONS = [
    1,
    3,
    5,
    10,
    20
]


MAX_HORIZON = max(
    HORIZONS
)


BATCH_SIZE = 64


TEST_EPISODES = range(
    180,
    200
)


# ============================================================
# 2. Rollout Dataset
#
# Each sample:
#
#   o_t
#
#   a_t ... a_(t+Hmax-1)
#
#   o_(t+1)
#   ...
#   o_(t+Hmax)
#
# Only windows that stay inside one episode are used.
# ============================================================

class ReacherRolloutDataset(
    Dataset
):

    def __init__(
        self,
        npz_path,
        episode_indices,
        max_horizon=20
    ):

        data = np.load(
            npz_path
        )


        self.frames = data[
            "frames"
        ]

        self.actions = data[
            "actions"
        ]


        episode_lengths = data[
            "episode_lengths"
        ]

        frame_offsets = data[
            "frame_offsets"
        ]

        action_offsets = data[
            "action_offsets"
        ]


        self.max_horizon = (
            max_horizon
        )


        # Each item:
        #
        # (
        #   frame_start,
        #   action_start
        # )

        self.samples = []


        for ep in episode_indices:

            length = int(
                episode_lengths[ep]
            )

            frame_start = int(
                frame_offsets[ep]
            )

            action_start = int(
                action_offsets[ep]
            )


            # Need:
            #
            # t + max_horizon <= length

            num_valid_starts = (
                length
                -
                max_horizon
                +
                1
            )


            for t in range(
                num_valid_starts
            ):

                self.samples.append(
                    (
                        frame_start + t,
                        action_start + t
                    )
                )


        print(
            f"Rollout samples: "
            f"{len(self.samples)}"
        )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        idx
    ):

        (
            frame_start,
            action_start
        ) = self.samples[idx]


        # ========================================
        # Initial frame
        # ========================================

        frame_0 = (
            preprocess_reacher_frame(
                self.frames[
                    frame_start
                ]
            )
        )


        # ========================================
        # Action sequence
        #
        # shape:
        # (Hmax, 2)
        # ========================================

        actions = (
            self.actions[
                action_start:
                action_start
                +
                self.max_horizon
            ]
        )


        actions = torch.from_numpy(
            actions.astype(
                np.float32
            )
        )


        # ========================================
        # Ground-truth future frames
        #
        # frame_start + 1
        # ...
        # frame_start + Hmax
        # ========================================

        future_frames = []


        for h in range(
            1,
            self.max_horizon + 1
        ):

            frame_h = (
                preprocess_reacher_frame(
                    self.frames[
                        frame_start + h
                    ]
                )
            )


            future_frames.append(
                frame_h
            )


        future_frames = torch.stack(
            future_frames,
            dim=0
        )


        return (
            frame_0,
            actions,
            future_frames
        )


# ============================================================
# 3. Dataset / Loader
# ============================================================

test_dataset = (
    ReacherRolloutDataset(
        DATA_PATH,
        TEST_EPISODES,
        max_horizon=MAX_HORIZON
    )
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    "Test rollout samples:",
    len(test_dataset)
)


# ============================================================
# 4. Load visual model
# ============================================================

autoencoder = VisualAutoencoder(
    latent_dim=LATENT_DIM
).to(
    device
)


autoencoder.load_state_dict(
    torch.load(
        AE_CHECKPOINT,
        map_location=device
    ),
    strict=False
)


for p in autoencoder.parameters():

    p.requires_grad = False


autoencoder.eval()


# ============================================================
# 5. Load Action Dynamics
# ============================================================

action_dynamics = LatentDynamics(
    latent_dim=LATENT_DIM,
    action_dim=ACTION_DIM,
    hidden_dim=HIDDEN_DIM
).to(
    device
)


action_dynamics.load_state_dict(
    torch.load(
        ACTION_CHECKPOINT,
        map_location=device
    )
)


action_dynamics.eval()


# ============================================================
# 6. Load No-action Dynamics
# ============================================================

noaction_dynamics = (
    NoActionLatentDynamics(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(
        device
    )
)


noaction_dynamics.load_state_dict(
    torch.load(
        NOACTION_CHECKPOINT,
        map_location=device
    )
)


noaction_dynamics.eval()


# ============================================================
# 7. Per-sample MSE
# ============================================================

def per_sample_mse(
    prediction,
    target
):

    error = (
        prediction
        -
        target
    ) ** 2


    error = error.flatten(
        start_dim=1
    )


    return error.mean(
        dim=1
    )


# ============================================================
# 8. Storage
# ============================================================

results = {}


for h in HORIZONS:

    results[h] = {

        "identity_latent": 0.0,
        "noaction_latent": 0.0,
        "action_latent": 0.0,

        "identity_pixel": 0.0,
        "noaction_pixel": 0.0,
        "action_pixel": 0.0,

        "count": 0
    }


# ============================================================
# 9. Open-loop Evaluation
# ============================================================

with torch.no_grad():

    for (
        frame_0,
        actions,
        future_frames
    ) in test_loader:


        frame_0 = frame_0.to(
            device
        )


        actions = actions.to(
            device
        )


        future_frames = (
            future_frames.to(
                device
            )
        )


        B = frame_0.shape[0]


        # ========================================
        # Initial latent
        # ========================================

        z_0 = autoencoder.encoder(
            frame_0
        )


        # Identity never changes
        z_identity = z_0


        # Open-loop model states
        z_action = z_0.clone()

        z_noaction = z_0.clone()


        # ========================================
        # Roll forward
        # ========================================

        for step in range(
            1,
            MAX_HORIZON + 1
        ):

            action_t = actions[
                :,
                step - 1,
                :
            ]


            # ------------------------------------
            # Action-conditioned rollout
            # ------------------------------------

            z_action = (
                action_dynamics(
                    z_action,
                    action_t
                )
            )


            # ------------------------------------
            # No-action rollout
            # ------------------------------------

            z_noaction = (
                noaction_dynamics(
                    z_noaction
                )
            )


            # Only evaluate requested horizons

            if step not in HORIZONS:

                continue


            # ====================================
            # Ground truth at horizon h
            # ====================================

            gt_frame = (
                future_frames[
                    :,
                    step - 1
                ]
            )


            gt_z = (
                autoencoder.encoder(
                    gt_frame
                )
            )


            # ====================================
            # Latent MSE
            # ====================================

            identity_latent = (
                per_sample_mse(
                    z_identity,
                    gt_z
                )
            )


            noaction_latent = (
                per_sample_mse(
                    z_noaction,
                    gt_z
                )
            )


            action_latent = (
                per_sample_mse(
                    z_action,
                    gt_z
                )
            )


            # ====================================
            # Decode predictions
            # ====================================

            identity_frame = (
                autoencoder.decoder(
                    z_identity
                )
            )


            noaction_frame = (
                autoencoder.decoder(
                    z_noaction
                )
            )


            action_frame = (
                autoencoder.decoder(
                    z_action
                )
            )


            # ====================================
            # Pixel MSE
            # ====================================

            identity_pixel = (
                per_sample_mse(
                    identity_frame,
                    gt_frame
                )
            )


            noaction_pixel = (
                per_sample_mse(
                    noaction_frame,
                    gt_frame
                )
            )


            action_pixel = (
                per_sample_mse(
                    action_frame,
                    gt_frame
                )
            )


            # ====================================
            # Accumulate
            # ====================================

            results[
                step
            ][
                "identity_latent"
            ] += identity_latent.sum().item()


            results[
                step
            ][
                "noaction_latent"
            ] += noaction_latent.sum().item()


            results[
                step
            ][
                "action_latent"
            ] += action_latent.sum().item()


            results[
                step
            ][
                "identity_pixel"
            ] += identity_pixel.sum().item()


            results[
                step
            ][
                "noaction_pixel"
            ] += noaction_pixel.sum().item()


            results[
                step
            ][
                "action_pixel"
            ] += action_pixel.sum().item()


            results[
                step
            ][
                "count"
            ] += B


# ============================================================
# 10. Average
# ============================================================

for h in HORIZONS:

    count = results[h][
        "count"
    ]


    for key in [
        "identity_latent",
        "noaction_latent",
        "action_latent",
        "identity_pixel",
        "noaction_pixel",
        "action_pixel"
    ]:

        results[h][key] /= count


# ============================================================
# 11. Print Table
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "OPEN-LOOP LATENT DYNAMICS"
)

print(
    "=" * 120
)


print(
    f"{'H':>4} | "
    f"{'Identity Latent':>16} | "
    f"{'NoAction Latent':>16} | "
    f"{'Action Latent':>16} | "
    f"{'Action vs NoAct':>15}"
)


print(
    "-" * 120
)


for h in HORIZONS:

    identity = results[h][
        "identity_latent"
    ]

    noaction = results[h][
        "noaction_latent"
    ]

    action = results[h][
        "action_latent"
    ]


    gain = (
        (
            noaction
            -
            action
        )
        /
        noaction
        *
        100.0
    )


    print(
        f"{h:4d} | "
        f"{identity:16.8f} | "
        f"{noaction:16.8f} | "
        f"{action:16.8f} | "
        f"{gain:+14.2f}%"
    )


print(
    "\n"
    + "=" * 120
)

print(
    "OPEN-LOOP PIXEL PREDICTION"
)

print(
    "=" * 120
)


print(
    f"{'H':>4} | "
    f"{'Identity Pixel':>16} | "
    f"{'NoAction Pixel':>16} | "
    f"{'Action Pixel':>16} | "
    f"{'Action vs NoAct':>15}"
)


print(
    "-" * 120
)


for h in HORIZONS:

    identity = results[h][
        "identity_pixel"
    ]

    noaction = results[h][
        "noaction_pixel"
    ]

    action = results[h][
        "action_pixel"
    ]


    gain = (
        (
            noaction
            -
            action
        )
        /
        noaction
        *
        100.0
    )


    print(
        f"{h:4d} | "
        f"{identity:16.8f} | "
        f"{noaction:16.8f} | "
        f"{action:16.8f} | "
        f"{gain:+14.2f}%"
    )


# ============================================================
# 12. Plot Latent Error vs Horizon
# ============================================================

h_values = HORIZONS


identity_latent = [
    results[h][
        "identity_latent"
    ]
    for h in HORIZONS
]


noaction_latent = [
    results[h][
        "noaction_latent"
    ]
    for h in HORIZONS
]


action_latent = [
    results[h][
        "action_latent"
    ]
    for h in HORIZONS
]


plt.figure(
    figsize=(7, 4)
)


plt.plot(
    h_values,
    identity_latent,
    marker="o",
    label="Identity"
)


plt.plot(
    h_values,
    noaction_latent,
    marker="o",
    label="No-action"
)


plt.plot(
    h_values,
    action_latent,
    marker="o",
    label="Action-conditioned"
)


plt.xlabel(
    "Rollout Horizon"
)

plt.ylabel(
    "Latent MSE"
)

plt.title(
    "Open-loop Latent Prediction"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 13. Plot Pixel Error vs Horizon
# ============================================================

identity_pixel = [
    results[h][
        "identity_pixel"
    ]
    for h in HORIZONS
]


noaction_pixel = [
    results[h][
        "noaction_pixel"
    ]
    for h in HORIZONS
]


action_pixel = [
    results[h][
        "action_pixel"
    ]
    for h in HORIZONS
]


plt.figure(
    figsize=(7, 4)
)


plt.plot(
    h_values,
    identity_pixel,
    marker="o",
    label="Identity"
)


plt.plot(
    h_values,
    noaction_pixel,
    marker="o",
    label="No-action"
)


plt.plot(
    h_values,
    action_pixel,
    marker="o",
    label="Action-conditioned"
)


plt.xlabel(
    "Rollout Horizon"
)

plt.ylabel(
    "Pixel MSE"
)

plt.title(
    "Open-loop Visual Prediction"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 14. Qualitative rollout
#
# One trajectory:
#
# columns:
# H = 1,3,5,10,20
#
# rows:
#
# Ground Truth
# Identity
# No-action
# Action-conditioned
# ============================================================

(
    frame_0,
    actions,
    future_frames
) = test_dataset[0]


frame_0 = (
    frame_0
    .unsqueeze(0)
    .to(device)
)


actions = (
    actions
    .unsqueeze(0)
    .to(device)
)


future_frames = (
    future_frames
    .to(device)
)


with torch.no_grad():

    z_0 = autoencoder.encoder(
        frame_0
    )


    z_action = z_0.clone()

    z_noaction = z_0.clone()

    z_identity = z_0.clone()


    qualitative = {}


    for step in range(
        1,
        MAX_HORIZON + 1
    ):

        a_t = actions[
            :,
            step - 1
        ]


        z_action = action_dynamics(
            z_action,
            a_t
        )


        z_noaction = noaction_dynamics(
            z_noaction
        )


        if step in HORIZONS:

            gt = future_frames[
                step - 1
            ]


            identity_img = (
                autoencoder.decoder(
                    z_identity
                )[0]
            )


            noaction_img = (
                autoencoder.decoder(
                    z_noaction
                )[0]
            )


            action_img = (
                autoencoder.decoder(
                    z_action
                )[0]
            )


            qualitative[
                step
            ] = (
                gt.cpu(),
                identity_img.cpu(),
                noaction_img.cpu(),
                action_img.cpu()
            )


# ============================================================
# 15. Visualize qualitative rollout
# ============================================================

fig, axes = plt.subplots(
    4,
    len(HORIZONS),
    figsize=(14, 8)
)


for col, h in enumerate(
    HORIZONS
):

    (
        gt,
        identity_img,
        noaction_img,
        action_img
    ) = qualitative[h]


    images = [
        gt,
        identity_img,
        noaction_img,
        action_img
    ]


    for row in range(
        4
    ):

        axes[
            row,
            col
        ].imshow(
            images[row]
            .permute(1, 2, 0)
            .clamp(0, 1)
        )


        axes[
            row,
            col
        ].axis(
            "off"
        )


    axes[
        0,
        col
    ].set_title(
        f"H={h}"
    )


axes[0, 0].set_ylabel(
    "GT"
)

axes[1, 0].set_ylabel(
    "Identity"
)

axes[2, 0].set_ylabel(
    "No-action"
)

axes[3, 0].set_ylabel(
    "Action"
)


plt.suptitle(
    "Open-loop Visual World Model Rollout"
)

plt.tight_layout()

plt.show()