from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy import ndimage
from torch.utils.data import (
    Dataset,
    DataLoader
)

from dataset_modified import (
    ReacherMotionTargetDataset
)

from model_modified import VisualAutoencoder


# ============================================================
# 1. Paths / Config
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


SAVE_PATH = (
    SCRIPT_DIR
    / "best_visual_autoencoder_worldtarget.pt"
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
# 2. Hyperparameters
# ============================================================

LATENT_DIM = 256

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3


ARM_WEIGHT = 5.0

# Raw Reacher target coordinates are small
# (roughly a few tenths of a meter),
# so this weight keeps the auxiliary loss relevant
# without overwhelming image reconstruction.
TARGET_COORD_WEIGHT = 5.0


FG_THRESHOLD = 0.08
FG_LOW_THRESHOLD = 0.035

FINAL_DILATION_ITER = 1

NUM_BACKGROUND_SAMPLES = 400


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
# 3. Base datasets
#
# Model input is still RGB only.
# target_xy_world is an auxiliary training label.
# ============================================================

train_base = (
    ReacherMotionTargetDataset(
        DATA_PATH,
        TRAIN_EPISODES
    )
)


val_base = (
    ReacherMotionTargetDataset(
        DATA_PATH,
        VAL_EPISODES
    )
)


test_base = (
    ReacherMotionTargetDataset(
        DATA_PATH,
        TEST_EPISODES
    )
)


print(
    "\nTrain transitions:",
    len(train_base)
)

print(
    "Val transitions:",
    len(val_base)
)

print(
    "Test transitions:",
    len(test_base)
)


# ============================================================
# 4. Estimate static background
#
# Train split only.
# ============================================================

def estimate_background(
    dataset,
    num_samples=400
):

    num_samples = min(
        num_samples,
        len(dataset)
    )


    indices = np.linspace(
        0,
        len(dataset) - 1,
        num_samples,
        dtype=np.int64
    )


    samples = []


    for idx in indices:

        (
            image,
            _,
            _,
            _
        ) = dataset[
            int(idx)
        ]


        samples.append(
            image
        )


    samples = torch.stack(
        samples,
        dim=0
    )


    background = (
        samples.median(
            dim=0
        ).values
    )


    print(
        "Background samples:",
        samples.shape
    )

    print(
        "Estimated background:",
        background.shape
    )


    return background


background = estimate_background(
    train_base,
    NUM_BACKGROUND_SAMPLES
)


# ============================================================
# 5. Foreground difference
# ============================================================

def compute_difference(
    image,
    background
):

    difference = torch.abs(
        image
        -
        background
    )


    difference = (
        difference.mean(
            dim=0
        )
    )


    return difference


# ============================================================
# 6. Largest non-border connected component
# ============================================================

def largest_non_border_component(
    binary_mask
):

    structure = np.ones(
        (3, 3),
        dtype=bool
    )


    connected = (
        ndimage.binary_closing(
            binary_mask,
            structure=structure,
            iterations=1
        )
    )


    labels, num_labels = (
        ndimage.label(
            connected,
            structure=structure
        )
    )


    if num_labels == 0:

        return np.zeros_like(
            binary_mask,
            dtype=bool
        )


    H, W = (
        binary_mask.shape
    )


    best_label = None
    best_area = 0


    for label_id in range(
        1,
        num_labels + 1
    ):

        ys, xs = np.where(
            labels == label_id
        )


        if len(
            xs
        ) == 0:

            continue


        touches_border = (
            ys.min() == 0
            or
            ys.max() == H - 1
            or
            xs.min() == 0
            or
            xs.max() == W - 1
        )


        if touches_border:

            continue


        area = len(
            xs
        )


        if area > best_area:

            best_area = area
            best_label = label_id


    if best_label is None:

        areas = np.bincount(
            labels.ravel()
        )


        if len(
            areas
        ) <= 1:

            return np.zeros_like(
                binary_mask,
                dtype=bool
            )


        areas[0] = 0


        best_label = int(
            areas.argmax()
        )


    return (
        labels == best_label
    )


# ============================================================
# 7. Arm-only mask
#
# Keep the arm-mask method that already worked well.
# ============================================================

def make_arm_mask_single(
    image,
    background
):

    difference = compute_difference(
        image,
        background
    )


    diff_np = (
        difference
        .cpu()
        .numpy()
    )


    foreground_high = (
        diff_np
        >
        FG_THRESHOLD
    )


    component = (
        largest_non_border_component(
            foreground_high
        )
    )


    structure = np.ones(
        (3, 3),
        dtype=bool
    )


    search_region = (
        ndimage.binary_dilation(
            component,
            structure=structure,
            iterations=1
        )
    )


    foreground_low = (
        diff_np
        >
        FG_LOW_THRESHOLD
    )


    arm_mask = (
        foreground_low
        &
        search_region
    )


    if (
        FINAL_DILATION_ITER
        >
        0
    ):

        arm_mask = (
            ndimage.binary_dilation(
                arm_mask,
                structure=structure,
                iterations=(
                    FINAL_DILATION_ITER
                )
            )
        )


    arm_mask = torch.from_numpy(
        arm_mask.astype(
            np.float32
        )
    ).unsqueeze(
        0
    )


    return arm_mask


# ============================================================
# 8. Dataset wrapper
#
# Precompute only the arm mask.
#
# Target label comes directly from simulator data.
# ============================================================

class ArmWorldTargetDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset,
        background,
        name="dataset"
    ):

        self.base_dataset = (
            base_dataset
        )


        masks = []


        print(
            f"\nPrecomputing "
            f"{name} arm masks..."
        )


        for i in range(
            len(base_dataset)
        ):

            (
                image,
                _,
                _,
                _
            ) = base_dataset[
                i
            ]


            arm_mask = (
                make_arm_mask_single(
                    image,
                    background
                )
            )


            masks.append(
                arm_mask.numpy().astype(
                    np.uint8
                )
            )


            if (
                (i + 1) % 500 == 0
                or
                i + 1
                ==
                len(base_dataset)
            ):

                print(
                    f"  {i + 1}/"
                    f"{len(base_dataset)}"
                )


        self.arm_masks = np.stack(
            masks,
            axis=0
        )


        print(
            f"{name} arm masks:",
            self.arm_masks.shape
        )


    def __len__(
        self
    ):

        return len(
            self.base_dataset
        )


    def __getitem__(
        self,
        idx
    ):

        (
            image,
            _,
            target_xy_world,
            _
        ) = self.base_dataset[
            idx
        ]


        arm_mask = torch.from_numpy(
            self.arm_masks[
                idx
            ]
        ).float()


        return (
            image,
            arm_mask,
            target_xy_world
        )


# ============================================================
# 9. Wrapped datasets / loaders
# ============================================================

train_dataset = ArmWorldTargetDataset(
    train_base,
    background,
    name="train"
)


val_dataset = ArmWorldTargetDataset(
    val_base,
    background,
    name="val"
)


test_dataset = ArmWorldTargetDataset(
    test_base,
    background,
    name="test"
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
# 10. Loss
# ============================================================

def compute_loss(
    recon,
    images,
    arm_mask,
    pred_target_xy_world,
    target_xy_world
):

    # ------------------------------------------
    # Global reconstruction
    # ------------------------------------------

    global_mse = F.mse_loss(
        recon,
        images
    )


    global_l1 = F.l1_loss(
        recon,
        images
    )


    global_loss = (
        global_mse
        +
        global_l1
    )


    # ------------------------------------------
    # Arm reconstruction
    # ------------------------------------------

    mask_rgb = arm_mask.expand_as(
        images
    )


    abs_error = torch.abs(
        recon
        -
        images
    )


    squared_error = (
        recon
        -
        images
    ) ** 2


    denominator = (
        mask_rgb.sum()
        .clamp_min(
            1.0
        )
    )


    arm_l1 = (
        abs_error
        *
        mask_rgb
    ).sum() / denominator


    arm_mse = (
        squared_error
        *
        mask_rgb
    ).sum() / denominator


    arm_loss = (
        arm_l1
        +
        arm_mse
    )


    # ------------------------------------------
    # Simulator target coordinate supervision
    #
    # IMPORTANT:
    # target_xy_world is not an input to encoder.
    # It is only an auxiliary label.
    # ------------------------------------------

    target_coord_loss = (
        F.mse_loss(
            pred_target_xy_world,
            target_xy_world
        )
    )


    # ------------------------------------------
    # Total
    # ------------------------------------------

    total_loss = (
        global_loss
        +
        ARM_WEIGHT
        *
        arm_loss
        +
        TARGET_COORD_WEIGHT
        *
        target_coord_loss
    )


    return (
        total_loss,
        global_loss,
        arm_loss,
        target_coord_loss,
        global_mse
    )


# ============================================================
# 11. World-coordinate error
# ============================================================

def target_world_error(
    pred_target_xy_world,
    target_xy_world
):

    error = torch.sqrt(
        (
            (
                pred_target_xy_world
                -
                target_xy_world
            ) ** 2
        ).sum(
            dim=1
        )
    )


    return error.mean()


# ============================================================
# 12. Model
# ============================================================

model = VisualAutoencoder(
    latent_dim=LATENT_DIM
).to(
    device
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_loss = float(
    "inf"
)


# ============================================================
# 13. Train
# ============================================================

print(
    "\n"
    + "=" * 110
)


print(
    "Training Arm-aware + Simulator-Target Visual Autoencoder"
)


print(
    "=" * 110
)


for epoch in range(
    NUM_EPOCHS
):

    # ========================================================
    # Train
    # ========================================================

    model.train()


    train_total = 0.0
    train_global = 0.0
    train_arm = 0.0
    train_target = 0.0
    train_target_err = 0.0


    for (
        images,
        arm_mask,
        target_xy_world
    ) in train_loader:


        images = images.to(
            device
        )


        arm_mask = arm_mask.to(
            device
        )


        target_xy_world = (
            target_xy_world.to(
                device
            )
        )


        (
            recon,
            z,
            pred_target_xy_world
        ) = model(
            images
        )


        (
            loss,
            global_loss,
            arm_loss,
            target_coord_loss,
            global_mse
        ) = compute_loss(
            recon,
            images,
            arm_mask,
            pred_target_xy_world,
            target_xy_world
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        world_error = (
            target_world_error(
                pred_target_xy_world.detach(),
                target_xy_world
            )
        )


        train_total += (
            loss.item()
        )

        train_global += (
            global_loss.item()
        )

        train_arm += (
            arm_loss.item()
        )

        train_target += (
            target_coord_loss.item()
        )

        train_target_err += (
            world_error.item()
        )


    train_total /= len(
        train_loader
    )

    train_global /= len(
        train_loader
    )

    train_arm /= len(
        train_loader
    )

    train_target /= len(
        train_loader
    )

    train_target_err /= len(
        train_loader
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()


    val_total = 0.0
    val_global = 0.0
    val_arm = 0.0
    val_target = 0.0
    val_target_err = 0.0


    with torch.no_grad():

        for (
            images,
            arm_mask,
            target_xy_world
        ) in val_loader:


            images = images.to(
                device
            )


            arm_mask = arm_mask.to(
                device
            )


            target_xy_world = (
                target_xy_world.to(
                    device
                )
            )


            (
                recon,
                z,
                pred_target_xy_world
            ) = model(
                images
            )


            (
                loss,
                global_loss,
                arm_loss,
                target_coord_loss,
                global_mse
            ) = compute_loss(
                recon,
                images,
                arm_mask,
                pred_target_xy_world,
                target_xy_world
            )


            world_error = (
                target_world_error(
                    pred_target_xy_world,
                    target_xy_world
                )
            )


            val_total += (
                loss.item()
            )

            val_global += (
                global_loss.item()
            )

            val_arm += (
                arm_loss.item()
            )

            val_target += (
                target_coord_loss.item()
            )

            val_target_err += (
                world_error.item()
            )


    val_total /= len(
        val_loader
    )

    val_global /= len(
        val_loader
    )

    val_arm /= len(
        val_loader
    )

    val_target /= len(
        val_loader
    )

    val_target_err /= len(
        val_loader
    )


    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Total {train_total:.6f} | "
        f"Global {train_global:.6f} | "
        f"Arm {train_arm:.6f} | "
        f"TargetMSE {train_target:.6f} | "
        f"TargetErr {train_target_err:.5f} || "
        f"Val Total {val_total:.6f} | "
        f"Global {val_global:.6f} | "
        f"Arm {val_arm:.6f} | "
        f"TargetMSE {val_target:.6f} | "
        f"TargetErr {val_target_err:.5f}"
    )


    if (
        val_total
        <
        best_val_loss
    ):

        best_val_loss = (
            val_total
        )


        torch.save(
            model.state_dict(),
            SAVE_PATH
        )


        print(
            "  -> Saved best model"
        )


# ============================================================
# 14. Load best model
# ============================================================

print(
    "\nLoading best model..."
)


model.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=device
    )
)


model.eval()


# ============================================================
# 15. Test
# ============================================================

test_total = 0.0
test_global = 0.0
test_arm = 0.0
test_target = 0.0
test_target_err = 0.0


all_true_targets = []
all_pred_targets = []


with torch.no_grad():

    for (
        images,
        arm_mask,
        target_xy_world
    ) in test_loader:


        images = images.to(
            device
        )


        arm_mask = arm_mask.to(
            device
        )


        target_xy_world = (
            target_xy_world.to(
                device
            )
        )


        (
            recon,
            z,
            pred_target_xy_world
        ) = model(
            images
        )


        (
            loss,
            global_loss,
            arm_loss,
            target_coord_loss,
            global_mse
        ) = compute_loss(
            recon,
            images,
            arm_mask,
            pred_target_xy_world,
            target_xy_world
        )


        world_error = (
            target_world_error(
                pred_target_xy_world,
                target_xy_world
            )
        )


        test_total += (
            loss.item()
        )

        test_global += (
            global_loss.item()
        )

        test_arm += (
            arm_loss.item()
        )

        test_target += (
            target_coord_loss.item()
        )

        test_target_err += (
            world_error.item()
        )


        all_true_targets.append(
            target_xy_world.cpu()
        )

        all_pred_targets.append(
            pred_target_xy_world.cpu()
        )


test_total /= len(
    test_loader
)

test_global /= len(
    test_loader
)

test_arm /= len(
    test_loader
)

test_target /= len(
    test_loader
)

test_target_err /= len(
    test_loader
)


all_true_targets = torch.cat(
    all_true_targets,
    dim=0
)

all_pred_targets = torch.cat(
    all_pred_targets,
    dim=0
)


print(
    "\n"
    + "=" * 80
)

print(
    "TEST RESULT"
)

print(
    "=" * 80
)

print(
    f"Total Loss        : "
    f"{test_total:.6f}"
)

print(
    f"Global Loss       : "
    f"{test_global:.6f}"
)

print(
    f"Arm Loss          : "
    f"{test_arm:.6f}"
)

print(
    f"Target Coord MSE  : "
    f"{test_target:.6f}"
)

print(
    f"Target World Error: "
    f"{test_target_err:.6f}"
)

print(
    "=" * 80
)


# ============================================================
# 16. Reconstruction visualization
# ============================================================

(
    images,
    arm_masks,
    target_xy_world
) = next(
    iter(test_loader)
)


images_device = (
    images.to(
        device
    )
)


with torch.no_grad():

    (
        recon,
        z,
        pred_target_xy_world
    ) = model(
        images_device
    )


recon = recon.cpu()


fig, axes = plt.subplots(
    3,
    8,
    figsize=(14, 6)
)


for j in range(
    8
):

    axes[0, j].imshow(
        images[j]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    axes[1, j].imshow(
        recon[j]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    axes[2, j].imshow(
        arm_masks[j, 0],
        cmap="gray",
        vmin=0,
        vmax=1
    )


    for row in range(
        3
    ):

        axes[
            row,
            j
        ].axis(
            "off"
        )


axes[0, 0].set_ylabel(
    "Ground Truth"
)

axes[1, 0].set_ylabel(
    "Reconstruction"
)

axes[2, 0].set_ylabel(
    "Arm Mask"
)


plt.suptitle(
    "Autoencoder Reconstruction"
)

plt.tight_layout()

plt.show()


# ============================================================
# 17. Target-world-coordinate visualization
#
# This visualization does NOT assume a camera-to-pixel mapping.
# It compares target positions directly in MuJoCo world space.
# ============================================================

num_show = min(
    200,
    len(
        all_true_targets
    )
)


plt.figure(
    figsize=(6, 6)
)


plt.scatter(
    all_true_targets[
        :num_show,
        0
    ].numpy(),
    all_true_targets[
        :num_show,
        1
    ].numpy(),
    marker="o",
    label="Ground Truth"
)


plt.scatter(
    all_pred_targets[
        :num_show,
        0
    ].numpy(),
    all_pred_targets[
        :num_show,
        1
    ].numpy(),
    marker="x",
    label="Prediction"
)


plt.xlabel(
    "Target world x"
)

plt.ylabel(
    "Target world y"
)

plt.title(
    "Target Coordinate Prediction"
)

plt.axis(
    "equal"
)

plt.legend()

plt.tight_layout()

plt.show()
