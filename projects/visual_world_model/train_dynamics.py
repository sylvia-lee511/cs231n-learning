from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset_modified import (
    ReacherTransitionDataset
)

from model import (
    VisualAutoencoder,
    LatentDynamics
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


DYNAMICS_CHECKPOINT = (
    SCRIPT_DIR
    / "best_latent_dynamics.pt"
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


# ============================================================
# Hyperparameters
# ============================================================

LATENT_DIM = 256
ACTION_DIM = 2
HIDDEN_DIM = 512

BATCH_SIZE = 128

NUM_EPOCHS = 50

LEARNING_RATE = 3e-4


TRAIN_EPISODES = range(
    0,
    160
)

VAL_EPISODES = range(
    160,
    180
)

TEST_EPISODES = range(
    180,
    200
)


# ============================================================
# 2. Dataset
# ============================================================

train_dataset = (
    ReacherTransitionDataset(
        DATA_PATH,
        TRAIN_EPISODES
    )
)


val_dataset = (
    ReacherTransitionDataset(
        DATA_PATH,
        VAL_EPISODES
    )
)


test_dataset = (
    ReacherTransitionDataset(
        DATA_PATH,
        TEST_EPISODES
    )
)


print()

print(
    "Train:",
    len(train_dataset)
)

print(
    "Val:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 3. Load pretrained visual model
# ============================================================

autoencoder = VisualAutoencoder(
    latent_dim=LATENT_DIM
).to(
    device
)


state_dict = torch.load(
    AE_CHECKPOINT,
    map_location=device
)


load_result = (
    autoencoder.load_state_dict(
        state_dict,
        strict=False
    )
)


print(
    "\nAutoencoder missing keys:",
    load_result.missing_keys
)

print(
    "Autoencoder unexpected keys:",
    load_result.unexpected_keys
)


# ============================================================
# Freeze visual model
# ============================================================

for parameter in (
    autoencoder.parameters()
):

    parameter.requires_grad = False


autoencoder.eval()


print(
    "Trainable AE params:",
    sum(
        p.numel()
        for p in autoencoder.parameters()
        if p.requires_grad
    )
)


# ============================================================
# 4. Dynamics
# ============================================================

dynamics = LatentDynamics(
    latent_dim=LATENT_DIM,
    action_dim=ACTION_DIM,
    hidden_dim=HIDDEN_DIM
).to(
    device
)


optimizer = torch.optim.Adam(
    dynamics.parameters(),
    lr=LEARNING_RATE
)


print(
    "Dynamics trainable params:",
    sum(
        p.numel()
        for p in dynamics.parameters()
        if p.requires_grad
    )
)


# ============================================================
# 5. Helper
# ============================================================

@torch.no_grad()
def encode_pair(
    frame_t,
    frame_next
):

    z_t = autoencoder.encoder(
        frame_t
    )


    z_next = autoencoder.encoder(
        frame_next
    )


    return (
        z_t,
        z_next
    )


# ============================================================
# 6. Validation
# ============================================================

@torch.no_grad()
def evaluate_latent(
    loader
):

    dynamics.eval()


    dynamics_mse = 0.0
    identity_mse = 0.0


    for (
        frame_t,
        action_t,
        frame_next
    ) in loader:


        frame_t = frame_t.to(
            device
        )

        action_t = action_t.to(
            device
        )

        frame_next = frame_next.to(
            device
        )


        (
            z_t,
            z_next
        ) = encode_pair(
            frame_t,
            frame_next
        )


        z_next_pred = dynamics(
            z_t,
            action_t
        )


        # ----------------------------------------
        # Learned dynamics
        # ----------------------------------------

        dyn_loss = F.mse_loss(
            z_next_pred,
            z_next
        )


        # ----------------------------------------
        # Identity baseline:
        #
        # simply predicts
        #
        # z_(t+1) = z_t
        # ----------------------------------------

        identity_loss = F.mse_loss(
            z_t,
            z_next
        )


        dynamics_mse += (
            dyn_loss.item()
        )

        identity_mse += (
            identity_loss.item()
        )


    dynamics_mse /= len(
        loader
    )

    identity_mse /= len(
        loader
    )


    return (
        dynamics_mse,
        identity_mse
    )


# ============================================================
# 7. Before training:
# measure identity baseline
# ============================================================

initial_dyn_mse, val_identity = (
    evaluate_latent(
        val_loader
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "INITIAL BASELINE"
)

print(
    "=" * 80
)

print(
    f"Initial dynamics MSE : "
    f"{initial_dyn_mse:.8f}"
)

print(
    f"Identity MSE         : "
    f"{val_identity:.8f}"
)

print(
    "=" * 80
)


# ============================================================
# 8. Train
# ============================================================

best_val_mse = float(
    "inf"
)


train_history = []
val_history = []


print(
    "\n"
    + "=" * 100
)

print(
    "Training Action-conditioned Latent Dynamics"
)

print(
    "=" * 100
)


for epoch in range(
    NUM_EPOCHS
):

    dynamics.train()


    train_mse = 0.0
    train_identity = 0.0


    for (
        frame_t,
        action_t,
        frame_next
    ) in train_loader:


        frame_t = frame_t.to(
            device
        )

        action_t = action_t.to(
            device
        )

        frame_next = frame_next.to(
            device
        )


        # ========================================
        # Frozen encoder
        # ========================================

        with torch.no_grad():

            z_t = (
                autoencoder.encoder(
                    frame_t
                )
            )

            z_next = (
                autoencoder.encoder(
                    frame_next
                )
            )


        # ========================================
        # Dynamics
        # ========================================

        z_next_pred = dynamics(
            z_t,
            action_t
        )


        loss = F.mse_loss(
            z_next_pred,
            z_next
        )


        # ========================================
        # Update dynamics only
        # ========================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ========================================
        # Logging
        # ========================================

        identity_loss = F.mse_loss(
            z_t,
            z_next
        )


        train_mse += (
            loss.item()
        )

        train_identity += (
            identity_loss.item()
        )


    train_mse /= len(
        train_loader
    )

    train_identity /= len(
        train_loader
    )


    # ==========================================
    # Validation
    # ==========================================

    (
        val_mse,
        val_identity
    ) = evaluate_latent(
        val_loader
    )


    train_history.append(
        train_mse
    )

    val_history.append(
        val_mse
    )


    # ==========================================
    # Improvement over identity
    # ==========================================

    if val_identity > 0:

        improvement = (
            (
                val_identity
                -
                val_mse
            )
            /
            val_identity
            *
            100.0
        )

    else:

        improvement = 0.0


    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train MSE {train_mse:.8f} | "
        f"Train Identity {train_identity:.8f} || "
        f"Val MSE {val_mse:.8f} | "
        f"Val Identity {val_identity:.8f} | "
        f"Improvement {improvement:+.2f}%"
    )


    # ==========================================
    # Save
    # ==========================================

    if val_mse < best_val_mse:

        best_val_mse = (
            val_mse
        )


        torch.save(
            dynamics.state_dict(),
            DYNAMICS_CHECKPOINT
        )


        print(
            "  -> Saved best dynamics"
        )


# ============================================================
# 9. Load best dynamics
# ============================================================

dynamics.load_state_dict(
    torch.load(
        DYNAMICS_CHECKPOINT,
        map_location=device
    )
)


dynamics.eval()


# ============================================================
# 10. Test latent metrics
# ============================================================

(
    test_dyn_mse,
    test_identity_mse
) = evaluate_latent(
    test_loader
)


test_improvement = (
    (
        test_identity_mse
        -
        test_dyn_mse
    )
    /
    test_identity_mse
    *
    100.0
)


print(
    "\n"
    + "=" * 80
)

print(
    "TEST LATENT DYNAMICS"
)

print(
    "=" * 80
)

print(
    f"Dynamics MSE : "
    f"{test_dyn_mse:.8f}"
)

print(
    f"Identity MSE : "
    f"{test_identity_mse:.8f}"
)

print(
    f"Improvement  : "
    f"{test_improvement:+.2f}%"
)

print(
    "=" * 80
)


# ============================================================
# 11. Visual one-step evaluation
# ============================================================

(
    frame_t,
    action_t,
    frame_next
) = next(
    iter(test_loader)
)


frame_t = frame_t.to(
    device
)

action_t = action_t.to(
    device
)

frame_next = frame_next.to(
    device
)


with torch.no_grad():

    z_t = autoencoder.encoder(
        frame_t
    )


    z_next_pred = dynamics(
        z_t,
        action_t
    )


    # Learned one-step prediction
    frame_next_pred = (
        autoencoder.decoder(
            z_next_pred
        )
    )


    # Identity baseline
    #
    # Decoder(z_t) is effectively:
    #
    # "next frame = current frame"

    frame_identity = (
        autoencoder.decoder(
            z_t
        )
    )


# ============================================================
# Pixel metrics
# ============================================================

prediction_pixel_mse = (
    F.mse_loss(
        frame_next_pred,
        frame_next
    ).item()
)


identity_pixel_mse = (
    F.mse_loss(
        frame_identity,
        frame_next
    ).item()
)


pixel_improvement = (
    (
        identity_pixel_mse
        -
        prediction_pixel_mse
    )
    /
    identity_pixel_mse
    *
    100.0
)


print(
    "\n"
    + "=" * 80
)

print(
    "ONE-STEP VISUAL PREDICTION"
)

print(
    "=" * 80
)

print(
    "Prediction pixel MSE:",
    prediction_pixel_mse
)

print(
    "Identity pixel MSE  :",
    identity_pixel_mse
)

print(
    f"Pixel improvement    : "
    f"{pixel_improvement:+.2f}%"
)

print(
    "=" * 80
)


# ============================================================
# 12. Visualization
#
# Row 1:
# Current frame o_t
#
# Row 2:
# Ground truth o_(t+1)
#
# Row 3:
# Identity baseline
#
# Row 4:
# Dynamics prediction
# ============================================================

frame_t = (
    frame_t.cpu()
)

frame_next = (
    frame_next.cpu()
)

frame_identity = (
    frame_identity.cpu()
)

frame_next_pred = (
    frame_next_pred.cpu()
)


fig, axes = plt.subplots(
    4,
    8,
    figsize=(14, 8)
)


for i in range(
    8
):

    axes[0, i].imshow(
        frame_t[i]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    axes[1, i].imshow(
        frame_next[i]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    axes[2, i].imshow(
        frame_identity[i]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    axes[3, i].imshow(
        frame_next_pred[i]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    for row in range(
        4
    ):

        axes[
            row,
            i
        ].axis(
            "off"
        )


axes[0, 0].set_ylabel(
    "Current"
)

axes[1, 0].set_ylabel(
    "GT Next"
)

axes[2, 0].set_ylabel(
    "Identity"
)

axes[3, 0].set_ylabel(
    "Dynamics"
)


plt.suptitle(
    "One-step Visual World Model Prediction"
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. Loss curve
# ============================================================

plt.figure(
    figsize=(7, 4)
)


plt.plot(
    train_history,
    label="Train Dynamics MSE"
)

plt.plot(
    val_history,
    label="Val Dynamics MSE"
)


plt.axhline(
    y=val_identity,
    linestyle="--",
    label="Val Identity Baseline"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Latent MSE"
)

plt.title(
    "Latent Dynamics Training"
)

plt.legend()

plt.tight_layout()

plt.show()