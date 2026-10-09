import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# 1. Pretrained ViT
# ==========================================

weights = ViT_B_16_Weights.DEFAULT

encoder = vit_b_16(
    weights=weights
)


print(encoder.heads)


# ==========================================
# 2. Remove classification head
# ==========================================

# TODO 1
#
# 把原来的 ImageNet classifier
# 替换成 Identity

encoder.heads = nn.Identity()


# ==========================================
# 3. Freeze encoder
# ==========================================

# TODO 2
#
# 所有 parameters:
# requires_grad = False

for param in encoder.parameters():
    param.requires_grad = False


encoder = encoder.to(device)

encoder.eval()

transform = weights.transforms()



dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)




loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False
)



all_features = []
all_labels = []

max_samples = 2000
count = 0


encoder.eval()

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        # TODO 1
        # pretrained ViT feature
        features = encoder(images)

        all_features.append(
            features.cpu()
        )

        all_labels.append(
            labels
        )

        count += images.size(0)

        if count >= max_samples:
            break


features = torch.cat(
    all_features,
    dim=0
)[:max_samples]

labels = torch.cat(
    all_labels,
    dim=0
)[:max_samples]


print("Features:", features.shape)
print("Labels:", labels.shape)




# ==========================================
# Normalize features
# ==========================================

# TODO 1
# 对 feature dimension 做 L2 normalization

features_norm = F.normalize(features, p=2, dim=1)


# ==========================================
# Cosine similarity matrix
# ==========================================

# TODO 2
#
# (B,768)
# @
# (768,B)
#
# ->
# (B,B)

similarity = features_norm @ features_norm.T



same_class = (
    labels[:, None]
    ==
    labels[None, :]
)

eye = torch.eye(
    len(labels),
    dtype=torch.bool
)

valid_pairs = ~eye


same_mask = (
    same_class & valid_pairs
)

diff_mask = (
    (~same_class) & valid_pairs
)



# TODO 4

same_similarity = similarity[same_mask].mean()

# TODO 5

diff_similarity = similarity[diff_mask].mean()


print(
    "Same-class similarity:",
    same_similarity.item()
)

print(
    "Different-class similarity:",
    diff_similarity.item()
)

print(
    "Similarity gap:",
    (
        same_similarity
        -
        diff_similarity
    ).item()
)


similarity_knn = similarity.clone()

similarity_knn.fill_diagonal_(
    -float("inf")
)

# TODO 6
# 每一行找到最相似图片的 index

nearest_indices = similarity_knn.argmax(
    dim=1
)


# TODO 7
# 根据 index 找到 nearest image 的 label

predicted_labels = labels[nearest_indices]


# TODO 8

knn_accuracy = (predicted_labels == labels).float().mean()


print(
    "1-NN Accuracy:",
    knn_accuracy.item()
)