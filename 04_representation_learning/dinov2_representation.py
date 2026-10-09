import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

model = model.to(device)
model.eval()

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

images, labels = next(
    iter(loader)
)

images = images.to(device)


with torch.no_grad():

    features = model(images)


print(
    "Images:",
    images.shape
)

print(
    "Features:",
    features.shape
)

all_features = []
all_labels = []

count = 0
max_samples = 2000


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        # TODO 1
        features = model(images)

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

features=F.normalize(
    features,
    dim=1
)

similarity_matrix = torch.mm(
    features,
    features.t()
)

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

same_similarity = similarity_matrix[same_mask].mean()

# TODO 5

diff_similarity = similarity_matrix[diff_mask].mean()


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


similarity_knn = similarity_matrix.clone()

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