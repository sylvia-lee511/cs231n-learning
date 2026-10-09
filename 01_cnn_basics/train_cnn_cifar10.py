import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

BATCH_SIZE=64

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:",device)

transform = transforms.ToTensor()

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
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class CNNEncoder(nn.Module):
    def __init__(self,feature_dim=256):
        super().__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(
                3,32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fn=nn.Linear(128*4*4,feature_dim)

    def forward(self,x):
        feature=self.encoder(x)
        feature = self.fn(feature)
        return feature


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder=CNNEncoder(feature_dim=256)
        self.classifier=nn.Linear(256,10)

    def forward(self,x):
        feature=self.encoder(x)
        logits=self.classifier(feature)

        return logits


classifier=Classifier().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=1e-3
)


num_epochs=20

for epoch in range(num_epochs):
    classifier.train()

    total_loss=0
    correct=0
    total=0

    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)

        logits=classifier(images)

        loss=criterion(logits,labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

        predictions=logits.argmax(dim=1)

        correct+=(predictions==labels).sum().item()

        total+=labels.size(0)

    train_accuracy=correct/total

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss {total_loss / len(train_loader):.4f} | "
        f"Accuracy {train_accuracy:.4f}"
    )

classifier.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = classifier(images)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

test_accuracy = correct / total

print(
    f"\nTest Accuracy: {test_accuracy:.4f}"
)