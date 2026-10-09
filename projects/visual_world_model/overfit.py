import torch
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    Subset
)

import matplotlib.pyplot as plt

from dataset import ReacherFrameDataset
from model import VisualAutoencoder


# ============================================================
# 1. Config
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


DATA_PATH = (
    "./data_visual_world_model/"
    "reacher_visual_random.npz"
)


LATENT_DIM = 128

TINY_NUM_SAMPLES = 64
BATCH_SIZE = 16

NUM_EPOCHS = 500

LEARNING_RATE = 1e-3


# foreground detection threshold
#
# 当前假设 image pixel range = [0, 1]
#
# 如果 mask 太少:
# 0.08 -> 0.05
#
# 如果 mask 太多:
# 0.08 -> 0.10 / 0.12
FOREGROUND_THRESHOLD = 0.08


# foreground loss 权重
FOREGROUND_WEIGHT = 5.0


# 是否对 foreground mask 膨胀
#
# 机械臂很细，建议开启
DILATE_MASK = True

MASK_KERNEL_SIZE = 5


# ============================================================
# 2. Dataset
# ============================================================

train_dataset = ReacherFrameDataset(
    DATA_PATH,
    episode_indices=range(0, 160)
)

val_dataset = ReacherFrameDataset(
    DATA_PATH,
    episode_indices=range(160, 180)
)

test_dataset = ReacherFrameDataset(
    DATA_PATH,
    episode_indices=range(180, 200)
)


print(
    "Train frames:",
    len(train_dataset)
)

print(
    "Val frames:",
    len(val_dataset)
)

print(
    "Test frames:",
    len(test_dataset)
)


# ============================================================
# 3. Tiny overfit dataset
# ============================================================

tiny_dataset = Subset(
    train_dataset,
    range(TINY_NUM_SAMPLES)
)


tiny_loader = DataLoader(
    tiny_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


print(
    "Tiny samples:",
    len(tiny_dataset)
)

print(
    "Updates per epoch:",
    len(tiny_loader)
)


# ============================================================
# 4. Estimate static background
#
# 对大量 frame 做 pixel-wise median
#
# moving arm / target 会被削弱
# static background 会留下
# ============================================================

background_samples = []


# 每隔10张取一张
for i in range(
    0,
    len(train_dataset),
    10
):

    background_samples.append(
        train_dataset[i]
    )


background_samples = torch.stack(
    background_samples,
    dim=0
)


print(
    "Background sample tensor:",
    background_samples.shape
)


# (N, 3, H, W)
# ->
# (3, H, W)

background = background_samples.median(
    dim=0
).values


print(
    "Background:",
    background.shape
)


# ============================================================
# 5. Visualize estimated background
# ============================================================

plt.figure(
    figsize=(4, 4)
)

plt.imshow(
    background
    .permute(1, 2, 0)
    .clamp(0, 1)
)

plt.title(
    "Estimated Static Background"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 6. Foreground mask function
# ============================================================

def make_foreground_mask(
    images,
    background,
    threshold=0.08
):

    """
    images:
        (B, 3, H, W)

    background:
        (3, H, W)

    return:
        mask:
        (B, 1, H, W)
    """

    # ------------------------------------------
    # 与背景做差
    # ------------------------------------------

    difference = torch.abs(
        images
        -
        background.unsqueeze(0)
    )


    # RGB channel 求平均
    #
    # (B,3,H,W)
    # ->
    # (B,1,H,W)

    difference = difference.mean(
        dim=1,
        keepdim=True
    )


    # ------------------------------------------
    # Binary foreground mask
    # ------------------------------------------

    mask = (
        difference
        >
        threshold
    ).float()


    # ------------------------------------------
    # Dilate mask
    #
    # 机械臂非常细，
    # 对 mask 稍微膨胀，
    # 让边缘附近也获得较高权重
    # ------------------------------------------

    if DILATE_MASK:

        padding = (
            MASK_KERNEL_SIZE // 2
        )

        mask = F.max_pool2d(
            mask,
            kernel_size=MASK_KERNEL_SIZE,
            stride=1,
            padding=padding
        )


    return mask


# ============================================================
# 7. Visualize mask BEFORE training
# ============================================================

preview_loader = DataLoader(
    tiny_dataset,
    batch_size=8,
    shuffle=False
)


preview_images = next(
    iter(preview_loader)
)


preview_masks = make_foreground_mask(
    preview_images,
    background,
    FOREGROUND_THRESHOLD
)


foreground_ratio = (
    preview_masks.mean().item()
)


print(
    "Preview foreground ratio:",
    foreground_ratio
)


fig, axes = plt.subplots(
    2,
    8,
    figsize=(14, 4)
)


for i in range(8):

    # Original
    axes[0, i].imshow(
        preview_images[i]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )

    axes[0, i].axis("off")


    # Mask
    axes[1, i].imshow(
        preview_masks[i, 0],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1, i].axis("off")


axes[0, 0].set_ylabel(
    "Input"
)

axes[1, 0].set_ylabel(
    "Mask"
)


plt.suptitle(
    "Foreground Mask Diagnostic"
)

plt.tight_layout()

plt.show()


# ============================================================
# 8. Model
# ============================================================

model = VisualAutoencoder(
    latent_dim=LATENT_DIM
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


background = background.to(
    device
)


# ============================================================
# 9. Training
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Foreground-weighted overfit test"
)

print(
    "=" * 70
)


for epoch in range(
    NUM_EPOCHS
):

    model.train()


    epoch_total_loss = 0.0
    epoch_global_loss = 0.0
    epoch_fg_loss = 0.0
    epoch_fg_ratio = 0.0


    for images in tiny_loader:

        images = images.to(
            device
        )


        # ========================================
        # Forward
        # ========================================

        recon, z = model(
            images
        )


        # ========================================
        # Foreground mask
        # ========================================

        mask = make_foreground_mask(
            images,
            background,
            FOREGROUND_THRESHOLD
        )


        # ========================================
        # Global reconstruction loss
        # ========================================

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


        # ========================================
        # Foreground reconstruction loss
        #
        # 这里只对 foreground pixel 求平均
        #
        # 不让背景 pixel 进入 denominator
        # ========================================

        mask_rgb = mask.expand_as(
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


        mask_sum = (
            mask_rgb.sum()
            .clamp_min(1.0)
        )


        foreground_l1 = (
            abs_error
            *
            mask_rgb
        ).sum() / mask_sum


        foreground_mse = (
            squared_error
            *
            mask_rgb
        ).sum() / mask_sum


        foreground_loss = (
            foreground_l1
            +
            foreground_mse
        )


        # ========================================
        # Total
        # ========================================

        loss = (
            global_loss
            +
            FOREGROUND_WEIGHT
            *
            foreground_loss
        )


        # ========================================
        # Update
        # ========================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ========================================
        # Logging
        # ========================================

        epoch_total_loss += (
            loss.item()
        )

        epoch_global_loss += (
            global_loss.item()
        )

        epoch_fg_loss += (
            foreground_loss.item()
        )

        epoch_fg_ratio += (
            mask.mean().item()
        )


    epoch_total_loss /= len(
        tiny_loader
    )

    epoch_global_loss /= len(
        tiny_loader
    )

    epoch_fg_loss /= len(
        tiny_loader
    )

    epoch_fg_ratio /= len(
        tiny_loader
    )


    if (
        epoch == 0
        or
        (epoch + 1) % 20 == 0
    ):

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Total {epoch_total_loss:.6f} | "
            f"Global {epoch_global_loss:.6f} | "
            f"Foreground {epoch_fg_loss:.6f} | "
            f"FG ratio {epoch_fg_ratio:.4f}"
        )


# ============================================================
# 10. Evaluation
# ============================================================

model.eval()


# 用固定顺序，不 shuffle
eval_loader = DataLoader(
    tiny_dataset,
    batch_size=8,
    shuffle=False
)


images = next(
    iter(eval_loader)
)


images = images.to(
    device
)


with torch.no_grad():

    recon, z = model(
        images
    )

    masks = make_foreground_mask(
        images,
        background,
        FOREGROUND_THRESHOLD
    )


# ============================================================
# 11. Final numerical diagnostic
# ============================================================

mask_rgb = masks.expand_as(
    images
)


global_test_mse = F.mse_loss(
    recon,
    images
)


foreground_test_mse = (
    (
        (recon - images) ** 2
        *
        mask_rgb
    ).sum()
    /
    mask_rgb.sum().clamp_min(1.0)
)


background_mask_rgb = (
    1.0
    -
    masks
).expand_as(
    images
)


background_test_mse = (
    (
        (recon - images) ** 2
        *
        background_mask_rgb
    ).sum()
    /
    background_mask_rgb
    .sum()
    .clamp_min(1.0)
)


print(
    "\n"
    + "=" * 70
)

print(
    "Final diagnostics"
)

print(
    "=" * 70
)

print(
    "Global MSE:",
    global_test_mse.item()
)

print(
    "Foreground MSE:",
    foreground_test_mse.item()
)

print(
    "Background MSE:",
    background_test_mse.item()
)

print(
    "Latent shape:",
    z.shape
)

print(
    "=" * 70
)


# ============================================================
# 12. Final visualization
#
# row 1: Ground Truth
# row 2: Reconstruction
# row 3: Foreground Mask
# ============================================================

images = images.cpu()

recon = recon.cpu()

masks = masks.cpu()


fig, axes = plt.subplots(
    3,
    8,
    figsize=(14, 6)
)


for i in range(8):

    # ------------------------------------------
    # Ground Truth
    # ------------------------------------------

    gt = images[i].permute(
        1, 2, 0
    )


    axes[0, i].imshow(
        gt.clamp(0, 1)
    )

    axes[0, i].axis("off")


    # ------------------------------------------
    # Reconstruction
    # ------------------------------------------

    pred = recon[i].permute(
        1, 2, 0
    )


    axes[1, i].imshow(
        pred.clamp(0, 1)
    )

    axes[1, i].axis("off")


    # ------------------------------------------
    # Mask
    # ------------------------------------------

    axes[2, i].imshow(
        masks[i, 0],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2, i].axis("off")


axes[0, 0].set_ylabel(
    "Ground Truth"
)

axes[1, 0].set_ylabel(
    "Reconstruction"
)

axes[2, 0].set_ylabel(
    "Foreground Mask"
)


plt.suptitle(
    "Tiny-set Overfit Diagnostic"
)

plt.tight_layout()

plt.show()