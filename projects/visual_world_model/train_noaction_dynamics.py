from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset_modified import (
    ReacherTransitionDataset
)

from model import (
    VisualAutoencoder,
    NoActionLatentDynamics
)


# ============================================================
# Config
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


SAVE_PATH = (
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
HIDDEN_DIM = 512

BATCH_SIZE = 128

# 之前 action model 大约 10~20 epoch 后
# validation 已基本饱和
NUM_EPOCHS = 25

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
# Dataset
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


# ============================================================
# Frozen Autoencoder
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


autoencoder.load_state_dict(
    state_dict,
    strict=False
)


for p in autoencoder.parameters():

    p.requires_grad = False


autoencoder.eval()


# ============================================================
# No-action dynamics
# ============================================================

dynamics = NoActionLatentDynamics(
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM
).to(
    device
)


optimizer = torch.optim.Adam(
    dynamics.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Encode
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
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    loader
):

    dynamics.eval()


    noaction_mse = 0.0
    identity_mse = 0.0


    for (
        frame_t,
        action_t,
        frame_next
    ) in loader:


        frame_t = frame_t.to(
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


        # ========================================
        # Important:
        #
        # action_t is intentionally NOT used
        # ========================================

        z_next_pred = dynamics(
            z_t
        )


        noaction_loss = F.mse_loss(
            z_next_pred,
            z_next
        )


        identity_loss = F.mse_loss(
            z_t,
            z_next
        )


        noaction_mse += (
            noaction_loss.item()
        )

        identity_mse += (
            identity_loss.item()
        )


    noaction_mse /= len(
        loader
    )

    identity_mse /= len(
        loader
    )


    return (
        noaction_mse,
        identity_mse
    )


# ============================================================
# Initial baseline
# ============================================================

initial_mse, identity_mse = evaluate(
    val_loader
)


print(
    "\n"
    + "=" * 70
)

print(
    "INITIAL NO-ACTION BASELINE"
)

print(
    "=" * 70
)

print(
    f"No-action MSE : "
    f"{initial_mse:.8f}"
)

print(
    f"Identity MSE  : "
    f"{identity_mse:.8f}"
)

print(
    "=" * 70
)


# ============================================================
# Train
# ============================================================

best_val_mse = float(
    "inf"
)


patience = 5
bad_epochs = 0


for epoch in range(
    NUM_EPOCHS
):

    dynamics.train()


    train_mse = 0.0


    for (
        frame_t,
        action_t,
        frame_next
    ) in train_loader:


        frame_t = frame_t.to(
            device
        )

        frame_next = frame_next.to(
            device
        )


        with torch.no_grad():

            z_t = autoencoder.encoder(
                frame_t
            )

            z_next = autoencoder.encoder(
                frame_next
            )


        # ----------------------------------------
        # No action input
        # ----------------------------------------

        z_next_pred = dynamics(
            z_t
        )


        loss = F.mse_loss(
            z_next_pred,
            z_next
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        train_mse += (
            loss.item()
        )


    train_mse /= len(
        train_loader
    )


    # ========================================================
    # Validation
    # ========================================================

    (
        val_mse,
        val_identity
    ) = evaluate(
        val_loader
    )


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


    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train MSE {train_mse:.8f} || "
        f"Val MSE {val_mse:.8f} | "
        f"Identity {val_identity:.8f} | "
        f"Improvement {improvement:+.2f}%"
    )


    # ========================================================
    # Best model + early stopping
    # ========================================================

    if val_mse < best_val_mse:

        best_val_mse = val_mse

        bad_epochs = 0


        torch.save(
            dynamics.state_dict(),
            SAVE_PATH
        )


        print(
            "  -> Saved best no-action dynamics"
        )

    else:

        bad_epochs += 1


        if bad_epochs >= patience:

            print(
                f"Early stopping at "
                f"epoch {epoch + 1}"
            )

            break


# ============================================================
# Load best
# ============================================================

dynamics.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=device
    )
)


dynamics.eval()


# ============================================================
# Test
# ============================================================

(
    test_noaction_mse,
    test_identity_mse
) = evaluate(
    test_loader
)


improvement_identity = (
    (
        test_identity_mse
        -
        test_noaction_mse
    )
    /
    test_identity_mse
    *
    100.0
)


# Current result from action-conditioned model
ACTION_DYNAMICS_MSE = (
    0.00191795
)


action_vs_noaction = (
    (
        test_noaction_mse
        -
        ACTION_DYNAMICS_MSE
    )
    /
    test_noaction_mse
    *
    100.0
)


print(
    "\n"
    + "=" * 80
)

print(
    "FINAL DYNAMICS COMPARISON"
)

print(
    "=" * 80
)


print(
    f"Identity MSE          : "
    f"{test_identity_mse:.8f}"
)

print(
    f"No-action MSE         : "
    f"{test_noaction_mse:.8f}"
)

print(
    f"Action-conditioned MSE: "
    f"{ACTION_DYNAMICS_MSE:.8f}"
)


print(
    "\nNo-action vs identity:"
)

print(
    f"{improvement_identity:+.2f}%"
)


print(
    "\nAction vs no-action:"
)

print(
    f"{action_vs_noaction:+.2f}%"
)


print(
    "=" * 80
)