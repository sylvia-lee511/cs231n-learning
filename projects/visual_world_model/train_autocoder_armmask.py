from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import (
    Dataset,
    DataLoader
)

from scipy import ndimage

from dataset import ReacherMotionFrameDataset
from model import VisualAutoencoder


# ============================================================
# 1. Paths / Config
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATA_PATH = (
    REPO_ROOT
    / "data_visual_world_model"
    / "reacher_visual_random.npz"
)

SAVE_PATH = (
    SCRIPT_DIR
    / "best_visual_autoencoder_armmask.pt"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Hyperparameters
# ============================================================

# 改成你当前 model.py 使用的 latent_dim
LATENT_DIM = 256

BATCH_SIZE = 64
NUM_EPOCHS = 30

LEARNING_RATE = 1e-3


# ============================================================
# Arm-mask parameters
# ============================================================

# 用于确定 foreground 连通区域
FG_THRESHOLD = 0.08

# 选出 arm 区域以后，
# 用更低阈值补回 anti-aliasing / 边缘像素
FG_LOW_THRESHOLD = 0.035


# 连通域构造前轻微 closing / dilation
CONNECT_DILATION_ITER = 1

# 最终 arm mask 再扩一圈
FINAL_DILATION_ITER = 1


# arm reconstruction 权重
ARM_WEIGHT = 5.0


# 用多少训练图片估计背景
NUM_BACKGROUND_SAMPLES = 400


# ============================================================
# Episode split
# ============================================================

TRAIN_EPISODES = range(0, 160)

VAL_EPISODES = range(160, 180)

TEST_EPISODES = range(180, 200)


# ============================================================
# 2. Base datasets
# ============================================================

train_base = ReacherMotionFrameDataset(
    DATA_PATH,
    TRAIN_EPISODES
)

val_base = ReacherMotionFrameDataset(
    DATA_PATH,
    VAL_EPISODES
)

test_base = ReacherMotionFrameDataset(
    DATA_PATH,
    TEST_EPISODES
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
# 3. Estimate static background
#
# IMPORTANT:
# 只使用 TRAIN SET
#
# background:
# (3,H,W)
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


    print(
        "\nEstimating static background..."
    )


    for idx in indices:

        image, _ = dataset[
            int(idx)
        ]

        samples.append(
            image
        )


    samples = torch.stack(
        samples,
        dim=0
    )


    print(
        "Background samples:",
        samples.shape
    )


    background = samples.median(
        dim=0
    ).values


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
# 4. Show estimated background
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

plt.axis(
    "off"
)

plt.tight_layout()

plt.show()


# ============================================================
# 5. Foreground difference
# ============================================================

def compute_difference(
    image,
    background
):

    """
    image:
        (3,H,W)

    background:
        (3,H,W)

    return:
        difference:
        (H,W)
    """

    difference = torch.abs(
        image
        -
        background
    )

    difference = difference.mean(
        dim=0
    )

    return difference


# ============================================================
# 6. Largest connected component
#
# 核心：
#
# foreground 中通常包括
#
#   arm
#   target
#
# arm 是最大的连通区域
# target 是较小独立圆点
#
# 同时忽略触碰图像边界的 component，
# 避免边框微小误差被选为 foreground。
# ============================================================

def largest_non_border_component(
    binary_mask
):

    """
    binary_mask:
        numpy bool array
        (H,W)

    return:
        bool mask
        (H,W)
    """


    # --------------------------------------------
    # 轻微 closing
    #
    # 把机械臂内部因为颜色变化造成的小裂缝连接起来
    # --------------------------------------------

    structure = np.ones(
        (3, 3),
        dtype=bool
    )


    connected = ndimage.binary_closing(
        binary_mask,
        structure=structure,
        iterations=1
    )


    # --------------------------------------------
    # 少量 dilation
    #
    # 保证两段机械臂/关节位置更容易连通
    # --------------------------------------------

#    if CONNECT_DILATION_ITER > 0:

#        connected = ndimage.binary_dilation(
#            connected,
#            structure=structure,
#            iterations=CONNECT_DILATION_ITER
#        )


    # --------------------------------------------
    # Connected components
    # --------------------------------------------

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


    H, W = binary_mask.shape


    best_label = None
    best_area = 0


    # --------------------------------------------
    # Find largest component
    #
    # 先排除触碰边界的 component
    # --------------------------------------------

    for label_id in range(
        1,
        num_labels + 1
    ):

        ys, xs = np.where(
            labels == label_id
        )


        if len(xs) == 0:

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


        area = len(xs)


        if area > best_area:

            best_area = area
            best_label = label_id


    # --------------------------------------------
    # 如果所有 component 都碰边界，
    # fallback:
    # 直接选最大 component
    # --------------------------------------------

    if best_label is None:

        areas = np.bincount(
            labels.ravel()
        )

        if len(areas) <= 1:

            return np.zeros_like(
                binary_mask,
                dtype=bool
            )


        areas[0] = 0

        best_label = int(
            areas.argmax()
        )


    component = (
        labels == best_label
    )


    return component


# ============================================================
# 7. Build precise arm-only mask
# ============================================================

def make_arm_mask_single(
    image,
    background
):

    """
    image:
        torch.Tensor
        (3,H,W)

    return:
        torch.Tensor
        (1,H,W)
    """


    difference = compute_difference(
        image,
        background
    )


    diff_np = (
        difference
        .cpu()
        .numpy()
    )


    # --------------------------------------------
    # High threshold:
    # 用于可靠确定连通区域
    # --------------------------------------------

    foreground_high = (
        diff_np
        >
        FG_THRESHOLD
    )


    # --------------------------------------------
    # 找最大的非背景连通域
    # --------------------------------------------

    component = (
        largest_non_border_component(
            foreground_high
        )
    )


    # --------------------------------------------
    # component 稍微扩大，
    # 再与 low-threshold foreground 求交
    #
    # 这样既保留精确轮廓，
    # 又能补回 anti-aliasing pixels
    # --------------------------------------------

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


    # --------------------------------------------
    # 最后只膨胀非常小的一圈
    #
    # 不再像 temporal motion mask 那样
    # 扩成一个大 blob
    # --------------------------------------------

    if FINAL_DILATION_ITER > 0:

        arm_mask = (
            ndimage.binary_dilation(
                arm_mask,
                structure=structure,
                iterations=FINAL_DILATION_ITER
            )
        )


    arm_mask = torch.from_numpy(
        arm_mask.astype(
            np.float32
        )
    )


    arm_mask = arm_mask.unsqueeze(
        0
    )


    return arm_mask


# ============================================================
# 8. Diagnostic:
# foreground vs selected arm mask
# ============================================================

def show_mask_diagnostic(
    dataset,
    background,
    num_images=8
):

    fig, axes = plt.subplots(
        3,
        num_images,
        figsize=(14, 6)
    )


    for i in range(
        num_images
    ):

        # 分散取样，
        # 不只看同一个 episode 开头
        idx = int(
            i
            *
            (len(dataset) - 1)
            /
            max(
                num_images - 1,
                1
            )
        )


        image, _ = dataset[idx]


        difference = compute_difference(
            image,
            background
        )


        foreground = (
            difference
            >
            FG_THRESHOLD
        ).float()


        arm_mask = make_arm_mask_single(
            image,
            background
        )


        # ----------------------------------------
        # Image
        # ----------------------------------------

        axes[0, i].imshow(
            image
            .permute(1, 2, 0)
            .clamp(0, 1)
        )


        # ----------------------------------------
        # Raw foreground
        # ----------------------------------------

        axes[1, i].imshow(
            foreground,
            cmap="gray",
            vmin=0,
            vmax=1
        )


        # ----------------------------------------
        # Final arm-only mask
        # ----------------------------------------

        axes[2, i].imshow(
            arm_mask[0],
            cmap="gray",
            vmin=0,
            vmax=1
        )


        for row in range(3):

            axes[row, i].axis(
                "off"
            )


    axes[0, 0].set_ylabel(
        "Image"
    )

    axes[1, 0].set_ylabel(
        "Foreground"
    )

    axes[2, 0].set_ylabel(
        "Arm Mask"
    )


    plt.suptitle(
        "Arm-mask Diagnostic"
    )

    plt.tight_layout()

    plt.show()


show_mask_diagnostic(
    train_base,
    background
)


# ============================================================
# 9. Precomputed Arm-mask Dataset
#
# Connected-components 不适合每个 epoch 都重新计算。
#
# 所以训练前一次性生成 mask。
# ============================================================

class ArmMaskDataset(Dataset):

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

            image, _ = (
                base_dataset[i]
            )


            arm_mask = (
                make_arm_mask_single(
                    image,
                    background
                )
            )


            # uint8 节省内存
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
            f"{name} masks:",
            self.arm_masks.shape
        )


    def __len__(self):

        return len(
            self.base_dataset
        )


    def __getitem__(
        self,
        idx
    ):

        image, _ = (
            self.base_dataset[idx]
        )


        arm_mask = torch.from_numpy(
            self.arm_masks[idx]
        ).float()


        return (
            image,
            arm_mask
        )


# ============================================================
# 10. Wrap datasets
# ============================================================

train_dataset = ArmMaskDataset(
    train_base,
    background,
    name="train"
)

val_dataset = ArmMaskDataset(
    val_base,
    background,
    name="val"
)

test_dataset = ArmMaskDataset(
    test_base,
    background,
    name="test"
)


# ============================================================
# 11. DataLoaders
# ============================================================

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



def compute_loss(
    recon,
    images,
    arm_mask
):

    # ==========================================
    # Global reconstruction
    # ==========================================

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


    # ==========================================
    # Arm reconstruction
    # ==========================================

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
        .clamp_min(1.0)
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


    # ==========================================
    # Target mask
    # ==========================================




    # ==========================================
    # Total
    # ==========================================

    total_loss = (
        global_loss
        +
        5.0 * arm_loss
    )


    return (
        total_loss,
        global_loss,
        arm_loss,
        global_mse
    )


# ============================================================
# 13. Model
# ============================================================

model = VisualAutoencoder(
    latent_dim=LATENT_DIM
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_loss = float(
    "inf"
)


# ============================================================
# 14. Train
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "Training Arm-aware Visual Autoencoder"
)

print(
    "=" * 100
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
    train_mse = 0.0


    for (
        images,
        arm_mask
    ) in train_loader:


        images = images.to(
            device
        )

        arm_mask = arm_mask.to(
            device
        )


        recon, z = model(
            images
        )


        (
            loss,
            global_loss,
            arm_loss,
            global_mse

        ) = compute_loss(
            recon,
            images,
            arm_mask
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        train_total += (
            loss.item()
        )

        train_global += (
            global_loss.item()
        )

        train_arm += (
            arm_loss.item()
        )

        train_mse += (
            global_mse.item()
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

    train_mse /= len(
        train_loader
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()


    val_total = 0.0
    val_global = 0.0
    val_arm = 0.0
    val_mse = 0.0


    with torch.no_grad():

        for (
            images,
            arm_mask
        ) in val_loader:


            images = images.to(
                device
            )

            arm_mask = arm_mask.to(
                device
            )


            recon, z = model(
                images
            )


            (
                loss,
                global_loss,
                arm_loss,
                global_mse
            ) = compute_loss(
                recon,
                images,
                arm_mask
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

            val_mse += (
                global_mse.item()
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

    val_mse /= len(
        val_loader
    )


    # ========================================================
    # Logging
    # ========================================================

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Total {train_total:.6f} | "
        f"Global {train_global:.6f} | "
        f"Arm {train_arm:.6f} | "
        f"MSE {train_mse:.6f} || "
        f"Val Total {val_total:.6f} | "
        f"Global {val_global:.6f} | "
        f"Arm {val_arm:.6f} | "
        f"MSE {val_mse:.6f}"
    )


    # ========================================================
    # Save best
    # ========================================================

    if val_total < best_val_loss:

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
# 15. Load best
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
# 16. Test
# ============================================================

test_total = 0.0
test_global = 0.0
test_arm = 0.0
test_mse = 0.0


with torch.no_grad():

    for (
        images,
        arm_mask
    ) in test_loader:


        images = images.to(
            device
        )

        arm_mask = arm_mask.to(
            device
        )


        recon, z = model(
            images
        )


        (
            loss,
            global_loss,
            arm_loss,
            global_mse
        ) = compute_loss(
            recon,
            images,
            arm_mask
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

        test_mse += (
            global_mse.item()
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

test_mse /= len(
    test_loader
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
    f"Total Loss : "
    f"{test_total:.6f}"
)

print(
    f"Global Loss: "
    f"{test_global:.6f}"
)

print(
    f"Arm Loss   : "
    f"{test_arm:.6f}"
)

print(
    f"Global MSE : "
    f"{test_mse:.6f}"
)

print(
    "=" * 80
)


# ============================================================
# 17. Final visualization
#
# Row 1:
# Ground Truth
#
# Row 2:
# Reconstruction
#
# Row 3:
# Arm Mask
# ============================================================

images, arm_masks = next(
    iter(test_loader)
)


images = images.to(
    device
)


with torch.no_grad():

    recon, z = model(
        images
    )


images = images.cpu()

recon = recon.cpu()


fig, axes = plt.subplots(
    3,
    8,
    figsize=(14, 6)
)


for i in range(56,64):

    # ----------------------------------------
    # GT
    # ----------------------------------------

    axes[0, i-56].imshow(
        images[i]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )

    axes[0, i-56].axis(
        "off"
    )


    # ----------------------------------------
    # Reconstruction
    # ----------------------------------------

    axes[1, i-56].imshow(
        recon[i]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )

    axes[1, i-56].axis(
        "off"
    )


    # ----------------------------------------
    # Arm mask
    # ----------------------------------------

    axes[2, i-56].imshow(
        arm_masks[i, 0],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2, i-56].axis(
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
    "Test Reconstruction - Arm Mask"
)

plt.tight_layout()

plt.show()