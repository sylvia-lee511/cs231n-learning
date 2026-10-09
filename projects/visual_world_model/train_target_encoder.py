import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset_modified import ReacherMotionTargetDataset
from model import TargetEncoder


# ============================================================
# Config
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


DATA_PATH = (
    "./data_visual_world_model/"
    "reacher_visual_random.npz"
)


BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 1e-3


# ============================================================
# Dataset
# ============================================================

train_dataset = ReacherMotionTargetDataset(
    DATA_PATH,
    range(0, 160)
)

val_dataset = ReacherMotionTargetDataset(
    DATA_PATH,
    range(160, 180)
)

test_dataset = ReacherMotionTargetDataset(
    DATA_PATH,
    range(180, 200)
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
# Model
# ============================================================

model = TargetEncoder().to(
    device
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# ============================================================
# Metric
# ============================================================

def world_error(
    pred,
    target
):

    error = torch.sqrt(
        (
            (
                pred
                -
                target
            ) ** 2
        ).sum(dim=1)
    )

    return error.mean()


# ============================================================
# Constant baseline
#
# 检查网络是否真的比
# "永远预测训练集平均target"
# 更好
# ============================================================

train_targets = torch.from_numpy(
    train_dataset.targets_t
).float()


mean_target = train_targets.mean(
    dim=0
)


test_targets = torch.from_numpy(
    test_dataset.targets_t
).float()


baseline_error = world_error(
    mean_target.unsqueeze(0).expand_as(
        test_targets
    ),
    test_targets
)


print(
    "Mean-target baseline error:",
    baseline_error.item()
)


# ============================================================
# Train
# ============================================================

best_val_error = float(
    "inf"
)


for epoch in range(
    NUM_EPOCHS
):

    model.train()

    train_loss = 0.0
    train_err = 0.0


    for (
        images,
        _,
        target_xy,
        _
    ) in train_loader:

        images = images.to(
            device
        )

        target_xy = target_xy.to(
            device
        )


        pred_xy = model(
            images
        )


        loss = F.mse_loss(
            pred_xy,
            target_xy
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
        )

        train_err += world_error(
            pred_xy.detach(),
            target_xy
        ).item()


    train_loss /= len(
        train_loader
    )

    train_err /= len(
        train_loader
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_err = 0.0


    with torch.no_grad():

        for (
            images,
            _,
            target_xy,
            _
        ) in val_loader:

            images = images.to(
                device
            )

            target_xy = target_xy.to(
                device
            )


            pred_xy = model(
                images
            )


            loss = F.mse_loss(
                pred_xy,
                target_xy
            )


            val_loss += (
                loss.item()
            )

            val_err += world_error(
                pred_xy,
                target_xy
            ).item()


    val_loss /= len(
        val_loader
    )

    val_err /= len(
        val_loader
    )


    print(
        f"Epoch {epoch+1:02d} | "
        f"Train MSE {train_loss:.6f} | "
        f"Train Err {train_err:.5f} || "
        f"Val MSE {val_loss:.6f} | "
        f"Val Err {val_err:.5f}"
    )


    if val_err < best_val_error:

        best_val_error = val_err

        torch.save(
            model.state_dict(),
            "best_target_encoder.pt"
        )


# ============================================================
# Test
# ============================================================

model.load_state_dict(
    torch.load(
        "best_target_encoder.pt",
        map_location=device
    )
)

model.eval()


test_loss = 0.0
test_err = 0.0


with torch.no_grad():

    for (
        images,
        _,
        target_xy,
        _
    ) in test_loader:

        images = images.to(
            device
        )

        target_xy = target_xy.to(
            device
        )


        pred_xy = model(
            images
        )


        test_loss += F.mse_loss(
            pred_xy,
            target_xy
        ).item()


        test_err += world_error(
            pred_xy,
            target_xy
        ).item()


test_loss /= len(
    test_loader
)

test_err /= len(
    test_loader
)


print()
print("=" * 60)
print("Target Encoder Test")
print("=" * 60)

print(
    "MSE:",
    test_loss
)

print(
    "World error:",
    test_err
)

print(
    "Mean baseline:",
    baseline_error.item()
)

print("=" * 60)