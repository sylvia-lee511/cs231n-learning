import torch
import torch.nn as nn

from torchvision import datasets
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights
)
from torch.utils.data import (
    DataLoader,
    TensorDataset
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


weights = ViT_B_16_Weights.DEFAULT

encoder = vit_b_16(
    weights=weights
)

encoder.heads = nn.Identity()


for param in encoder.parameters():
    param.requires_grad = False


encoder = encoder.to(device)
encoder.eval()


transform = weights.transforms()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

def extract_features(
    encoder,
    loader,
    device
):

    all_features = []
    all_labels = []

    encoder.eval()

    with torch.no_grad():

        for images, labels in loader:

            # TODO 1
            # images → GPU
            images=images.to(device)

            # TODO 2
            # feature extraction
            features = encoder(images)


            # TODO 3
            # append CPU features / labels
            all_features.append(features.cpu())
            all_labels.append(labels)

    # TODO 4
    # concatenate

    all_features = torch.cat(all_features,dim=0)
    all_labels = torch.cat(all_labels,dim=0)


    return all_features, all_labels


train_features, train_labels = extract_features(
    encoder,
    train_loader,
    device
)

test_features, test_labels = extract_features(
    encoder,
    test_loader,
    device
)


print(
    "Train:",
    train_features.shape,
    train_labels.shape
)

print(
    "Test:",
    test_features.shape,
    test_labels.shape
)

train_feature_dataset = TensorDataset(
    train_features,
    train_labels
)

test_feature_dataset = TensorDataset(
    test_features,
    test_labels
)

train_feature_loader = DataLoader(
    train_feature_dataset,
    batch_size=256,
    shuffle=True
)

test_feature_loader = DataLoader(
    test_feature_dataset,
    batch_size=256,
    shuffle=False
)

classifier = nn.Linear(
    768,
    10
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    classifier.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

encoder_trainable = sum(
    p.numel()
    for p in encoder.parameters()
    if p.requires_grad
)

classifier_trainable = sum(
    p.numel()
    for p in classifier.parameters()
    if p.requires_grad
)


print(
    "Encoder trainable:",
    encoder_trainable
)

print(
    "Classifier trainable:",
    classifier_trainable
)

NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):

    classifier.train()

    total_loss = 0.0
    correct = 0
    total = 0


    for features, labels in train_feature_loader:

        # TODO 5
        # move to GPU
        features = features.to(device)
        labels = labels.to(device)

        # TODO 6
        # zero grad
        optimizer.zero_grad()   

        # TODO 7
        # logits
        logits=classifier(features)

        # TODO 8
        # CE loss

        loss = criterion(logits, labels)

        # TODO 9
        # backward
        loss.backward()

        # TODO 10
        # optimizer step
        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = (
        total_loss
        /
        len(train_feature_loader)
    )

    train_acc = correct / total

    classifier.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for features, labels in test_feature_loader:

            # TODO 11
            # device
            features = features.to(device)
            labels = labels.to(device)

            # TODO 12
            # classifier
            logits = classifier(features)



            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    test_acc = correct / total


    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Test Acc {test_acc:.4f}"
    )