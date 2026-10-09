import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vit import VisionTransformer


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. Hyperparameters
# ============================================================

BATCH_SIZE = 128
NUM_EPOCHS = 20

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05


# ============================================================
# 3. Data augmentation / normalization
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = VisionTransformer(
    image_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,

    embed_dim=128,
    depth=4,
    num_heads=4,
    mlp_ratio=4.0,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

for epoch in range(NUM_EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        # TODO 1
        # move to GPU
        images = images.to(device)
        labels = labels.to(device)

        # TODO 2
        # clear gradients
        optimizer.zero_grad()

        # TODO 3
        # forward
        logits = model(images)


        # TODO 4
        # CrossEntropy
        loss = criterion(logits, labels)


        # TODO 5
        # backward
        loss.backward()

        if epoch == 0:

            print(
                "CLS grad:",
                model.embedding.cls_token.grad.norm()
            )

            print(
                "Position grad:",
                model.embedding.pos_embed.grad.norm()
            )

            print(
                "Patch projection grad:",
                model.embedding.patch_embed.proj.weight.grad.norm()
            )

        # TODO 6
        # update parameters
        optimizer.step()

        # ==========================================
        # Statistics
        # ==========================================

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = (
        total_loss / len(train_loader)
    )

    train_acc = correct / total


    # TODO 7
    # scheduler update
    scheduler.step()

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f}"
    )

    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            # TODO 8
            # GPU
            images = images.to(device)
            labels = labels.to(device)


            # TODO 9
            # forward
            logits = model(images)


            # TODO 10
            # loss
            loss = criterion(logits, labels)


            test_loss += loss.item()

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    test_loss /= len(test_loader)
    test_acc = correct / total


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"Epoch {epoch + 1:02d} | "
        f"LR {current_lr:.6f} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Test Loss {test_loss:.4f} | "
        f"Test Acc {test_acc:.4f}"
    )


torch.save(
    model.state_dict(),
    "tiny_vit_cifar10.pt"
)
