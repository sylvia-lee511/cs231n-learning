from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

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
    / "best_visual_autoencoder_targetcoord.pt"
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 2. Hyperparameters
# ============================================================

LATENT_DIM = 256

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

ARM_WEIGHT = 5.0
TARGET_COORD_WEIGHT = 5.0

FG_THRESHOLD = 0.08
FG_LOW_THRESHOLD = 0.035

FINAL_DILATION_ITER = 1

NUM_BACKGROUND_SAMPLES = 400

TRAIN_EPISODES = range(0, 160)
VAL_EPISODES = range(160, 180)
TEST_EPISODES = range(180, 200)


# ============================================================
# 3. Base datasets
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
# 4. Estimate static background
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


    background = samples.median(
        dim=0
    ).values


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
        image - background
    )

    difference = difference.mean(
        dim=0
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


    connected = ndimage.binary_closing(
        binary_mask,
        structure=structure,
        iterations=1
    )


    labels, num_labels = ndimage.label(
        connected,
        structure=structure
    )


    if num_labels == 0:

        return np.zeros_like(
            binary_mask,
            dtype=bool
        )


    H, W = binary_mask.shape

    best_label = None
    best_area = 0


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
            or ys.max() == H - 1
            or xs.min() == 0
            or xs.max() == W - 1
        )


        if touches_border:
            continue


        area = len(xs)


        if area > best_area:

            best_area = area
            best_label = label_id


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


    return (
        labels == best_label
    )


# ============================================================
# 7. Arm-only mask
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
        diff_np > FG_THRESHOLD
    )


    component = largest_non_border_component(
        foreground_high
    )


    structure = np.ones(
        (3, 3),
        dtype=bool
    )


    search_region = ndimage.binary_dilation(
        component,
        structure=structure,
        iterations=1
    )


    foreground_low = (
        diff_np > FG_LOW_THRESHOLD
    )


    arm_mask = (
        foreground_low
        &
        search_region
    )


    if FINAL_DILATION_ITER > 0:

        arm_mask = ndimage.binary_dilation(
            arm_mask,
            structure=structure,
            iterations=FINAL_DILATION_ITER
        )


    arm_mask = torch.from_numpy(
        arm_mask.astype(
            np.float32
        )
    ).unsqueeze(0)


    return arm_mask


# ============================================================
# 8. Target coordinate extraction
#
# Target pixels are used ONLY to generate a coordinate label.
# They are NOT used as a pixel reconstruction loss.
# ============================================================

def extract_target_xy_single(
    image,
    arm_mask
):

    """
    image:
        (3,H,W), [0,1]

    arm_mask:
        (1,H,W)

    return:
        target_xy:
        tensor([x_norm, y_norm])
    """

    _, H, W = image.shape


    r = image[0]
    g = image[1]
    b = image[2]


    # Red candidates
    red_candidate = (
        (r > 0.25)
        &
        (r > g * 1.3)
        &
        (r > b * 1.3)
    )


    # Keep only inner workspace to exclude the pink frame
    margin_h = int(
        0.15 * H
    )

    margin_w = int(
        0.15 * W
    )


    workspace = torch.zeros(
        (H, W),
        dtype=torch.bool
    )

    workspace[
        margin_h:H-margin_h,
        margin_w:W-margin_w
    ] = True


    # Exclude red joint pixels around the robot arm
    arm_exclusion = F.max_pool2d(
        arm_mask.unsqueeze(0),
        kernel_size=9,
        stride=1,
        padding=4
    ).squeeze(0).squeeze(0) > 0.5


    candidate = (
        red_candidate
        &
        workspace
        &
        (~arm_exclusion)
    )


    candidate_np = (
        candidate
        .cpu()
        .numpy()
    )


    # Connected components among remaining red pixels
    structure = np.ones(
        (3, 3),
        dtype=bool
    )

    labels, num_labels = ndimage.label(
        candidate_np,
        structure=structure
    )


    best_component = None
    best_area = 0


    for label_id in range(
        1,
        num_labels + 1
    ):

        ys, xs = np.where(
            labels == label_id
        )


        if len(xs) == 0:
            continue


        area = len(xs)


        if area > best_area:

            best_area = area
            best_component = (
                xs,
                ys
            )


    # Normal path: centroid of largest red component
    if best_component is not None:

        xs, ys = best_component

        x = float(
            xs.mean()
        )

        y = float(
            ys.mean()
        )


    # Fallback:
    # choose the strongest redness point inside workspace,
    # excluding the arm.
    else:

        redness = torch.relu(
            r - 0.5 * (g + b)
        )


        valid = (
            workspace
            &
            (~arm_exclusion)
        )


        score = redness.clone()

        score[
            ~valid
        ] = -1.0


        flat_index = int(
            score.argmax().item()
        )


        y = float(
            flat_index // W
        )

        x = float(
            flat_index % W
        )


    x_norm = x / (
        W - 1
    )

    y_norm = y / (
        H - 1
    )


    return torch.tensor(
        [
            x_norm,
            y_norm
        ],
        dtype=torch.float32
    )


# ============================================================
# 9. Diagnostic
# ============================================================

def show_label_diagnostic(
    dataset,
    background,
    num_images=8
):

    fig, axes = plt.subplots(
        2,
        num_images,
        figsize=(14, 4)
    )


    for i in range(
        num_images
    ):

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


        arm_mask = make_arm_mask_single(
            image,
            background
        )


        target_xy = extract_target_xy_single(
            image,
            arm_mask
        )


        _, H, W = image.shape


        target_x = (
            target_xy[0].item()
            *
            (W - 1)
        )

        target_y = (
            target_xy[1].item()
            *
            (H - 1)
        )


        axes[0, i].imshow(
            image
            .permute(1, 2, 0)
            .clamp(0, 1)
        )

        axes[0, i].scatter(
            [target_x],
            [target_y],
            marker="x",
            s=40
        )

        axes[1, i].imshow(
            arm_mask[0],
            cmap="gray",
            vmin=0,
            vmax=1
        )


        axes[0, i].axis("off")
        axes[1, i].axis("off")


    axes[0, 0].set_ylabel(
        "Target label"
    )

    axes[1, 0].set_ylabel(
        "Arm mask"
    )


    plt.suptitle(
        "Target-coordinate Label Diagnostic"
    )

    plt.tight_layout()
    plt.show()


show_label_diagnostic(
    train_base,
    background
)


# ============================================================
# 10. Precomputed training dataset
# ============================================================

class ArmTargetDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        background,
        name="dataset"
    ):

        self.base_dataset = (
            base_dataset
        )


        arm_masks = []
        target_coords = []


        print(
            f"\nPrecomputing "
            f"{name} labels..."
        )


        for i in range(
            len(base_dataset)
        ):

            image, _ = (
                base_dataset[i]
            )


            arm_mask = make_arm_mask_single(
                image,
                background
            )


            target_xy = (
                extract_target_xy_single(
                    image,
                    arm_mask
                )
            )


            arm_masks.append(
                arm_mask.numpy().astype(
                    np.uint8
                )
            )


            target_coords.append(
                target_xy.numpy()
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
            arm_masks,
            axis=0
        )


        self.target_coords = np.stack(
            target_coords,
            axis=0
        ).astype(
            np.float32
        )


        print(
            f"{name} arm masks:",
            self.arm_masks.shape
        )

        print(
            f"{name} target coords:",
            self.target_coords.shape
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


        target_xy = torch.from_numpy(
            self.target_coords[idx]
        ).float()


        return (
            image,
            arm_mask,
            target_xy
        )


# ============================================================
# 11. Wrapped datasets
# ============================================================

train_dataset = ArmTargetDataset(
    train_base,
    background,
    name="train"
)

val_dataset = ArmTargetDataset(
    val_base,
    background,
    name="val"
)

test_dataset = ArmTargetDataset(
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
# 12. Loss
# ============================================================

def compute_loss(
    recon,
    images,
    arm_mask,
    pred_target_xy,
    target_xy
):

    # Global reconstruction
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


    # Arm reconstruction
    mask_rgb = arm_mask.expand_as(
        images
    )


    abs_error = torch.abs(
        recon - images
    )


    squared_error = (
        recon - images
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


    # Target coordinate prediction
    target_coord_loss = F.mse_loss(
        pred_target_xy,
        target_xy
    )


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
# 13. Pixel coordinate error
# ============================================================

def target_pixel_error(
    pred_target_xy,
    target_xy,
    H,
    W
):

    dx = (
        pred_target_xy[:, 0]
        -
        target_xy[:, 0]
    ) * (
        W - 1
    )


    dy = (
        pred_target_xy[:, 1]
        -
        target_xy[:, 1]
    ) * (
        H - 1
    )


    error = torch.sqrt(
        dx ** 2
        +
        dy ** 2
    )


    return error.mean()


# ============================================================
# 14. Model
#
# Assumption:
#
# model(images) returns:
#
#   recon, z, pred_target_xy
#
# where pred_target_xy is normalized to [0,1]^2
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
# 15. Train
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "Training Arm-aware + Target-coordinate Visual Autoencoder"
)

print(
    "=" * 110
)


for epoch in range(
    NUM_EPOCHS
):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()


    train_total = 0.0
    train_global = 0.0
    train_arm = 0.0
    train_target = 0.0
    train_target_px = 0.0


    for (
        images,
        arm_mask,
        target_xy
    ) in train_loader:


        images = images.to(
            device
        )

        arm_mask = arm_mask.to(
            device
        )

        target_xy = target_xy.to(
            device
        )


        # Your modified model should return 3 values here.
        recon, z, pred_target_xy = model(
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
            pred_target_xy,
            target_xy
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        H = images.shape[-2]
        W = images.shape[-1]


        px_error = target_pixel_error(
            pred_target_xy.detach(),
            target_xy,
            H,
            W
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

        train_target_px += (
            px_error.item()
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

    train_target_px /= len(
        train_loader
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()


    val_total = 0.0
    val_global = 0.0
    val_arm = 0.0
    val_target = 0.0
    val_target_px = 0.0


    with torch.no_grad():

        for (
            images,
            arm_mask,
            target_xy
        ) in val_loader:


            images = images.to(
                device
            )

            arm_mask = arm_mask.to(
                device
            )

            target_xy = target_xy.to(
                device
            )


            recon, z, pred_target_xy = model(
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
                pred_target_xy,
                target_xy
            )


            H = images.shape[-2]
            W = images.shape[-1]


            px_error = target_pixel_error(
                pred_target_xy,
                target_xy,
                H,
                W
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

            val_target_px += (
                px_error.item()
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

    val_target_px /= len(
        val_loader
    )


    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Total {train_total:.6f} | "
        f"Global {train_global:.6f} | "
        f"Arm {train_arm:.6f} | "
        f"TargetCoord {train_target:.6f} | "
        f"TargetPx {train_target_px:.3f} || "
        f"Val Total {val_total:.6f} | "
        f"Global {val_global:.6f} | "
        f"Arm {val_arm:.6f} | "
        f"TargetCoord {val_target:.6f} | "
        f"TargetPx {val_target_px:.3f}"
    )


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
# 16. Load best
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
# 17. Test
# ============================================================

test_total = 0.0
test_global = 0.0
test_arm = 0.0
test_target = 0.0
test_target_px = 0.0


with torch.no_grad():

    for (
        images,
        arm_mask,
        target_xy
    ) in test_loader:


        images = images.to(
            device
        )

        arm_mask = arm_mask.to(
            device
        )

        target_xy = target_xy.to(
            device
        )


        recon, z, pred_target_xy = model(
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
            pred_target_xy,
            target_xy
        )


        H = images.shape[-2]
        W = images.shape[-1]


        px_error = target_pixel_error(
            pred_target_xy,
            target_xy,
            H,
            W
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

        test_target_px += (
            px_error.item()
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

test_target_px /= len(
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
    f"Total Loss       : "
    f"{test_total:.6f}"
)

print(
    f"Global Loss      : "
    f"{test_global:.6f}"
)

print(
    f"Arm Loss         : "
    f"{test_arm:.6f}"
)

print(
    f"Target Coord MSE : "
    f"{test_target:.6f}"
)

print(
    f"Target Pixel Err : "
    f"{test_target_px:.3f} px"
)

print(
    "=" * 80
)


# ============================================================
# 18. Final visualization
#
# Row 1: Ground Truth
# Row 2: Reconstruction
# Row 3: Arm Mask
# Row 4: Target coordinate prediction on original image
#
# Circle = GT target
# X      = predicted target
# ============================================================

images, arm_masks, target_xy = next(
    iter(test_loader)
)


images_device = images.to(
    device
)


with torch.no_grad():

    recon, z, pred_target_xy = model(
        images_device
    )


recon = recon.cpu()
pred_target_xy = pred_target_xy.cpu()


fig, axes = plt.subplots(
    4,
    8,
    figsize=(14, 8)
)


for j in range(8):

    gt = (
        images[j]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    pred = (
        recon[j]
        .permute(1, 2, 0)
        .clamp(0, 1)
    )


    H = images.shape[-2]
    W = images.shape[-1]


    gt_x = (
        target_xy[j, 0].item()
        *
        (W - 1)
    )

    gt_y = (
        target_xy[j, 1].item()
        *
        (H - 1)
    )


    pred_x = (
        pred_target_xy[j, 0].item()
        *
        (W - 1)
    )

    pred_y = (
        pred_target_xy[j, 1].item()
        *
        (H - 1)
    )


    axes[0, j].imshow(
        gt
    )

    axes[1, j].imshow(
        pred
    )

    axes[2, j].imshow(
        arm_masks[j, 0],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3, j].imshow(
        gt
    )


    axes[3, j].scatter(
        [gt_x],
        [gt_y],
        marker="o",
        s=45,
        facecolors="none"
    )

    axes[3, j].scatter(
        [pred_x],
        [pred_y],
        marker="x",
        s=45
    )


    for row in range(4):

        axes[row, j].axis(
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

axes[3, 0].set_ylabel(
    "Target XY"
)


plt.suptitle(
    "Autoencoder + Target Coordinate Prediction"
)

plt.tight_layout()

plt.show()
