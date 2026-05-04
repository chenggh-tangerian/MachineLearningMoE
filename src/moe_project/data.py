from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class ClusterSpec:
    center: torch.Tensor
    label: int


class ClusteredToyDataset(Dataset):
    """合成数据集：每个类别对应一个高斯簇，适合验证 K-Means 路由。"""

    def __init__(self, num_samples: int, input_dim: int, num_classes: int, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.empty(num_samples, input_dim)
        self.labels = torch.empty(num_samples, dtype=torch.long)

        class_centers = torch.randn(num_classes, input_dim, generator=generator) * 3.0
        for index in range(num_samples):
            label = int(torch.randint(0, num_classes, (1,), generator=generator).item())
            noise = torch.randn(input_dim, generator=generator) * 0.7
            self.features[index] = class_centers[label] + noise
            self.labels[index] = label

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class DigitsBenchmarkDataset(Dataset):
    """标准 Digits 分类基准，适合和其他模型做可比实验。"""

    def __init__(self, train: bool, seed: int = 42, split_ratio: float = 0.8):
        try:
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise RuntimeError("DigitsBenchmarkDataset requires scikit-learn to be installed") from exc

        digits = load_digits()
        features = torch.tensor(digits.data, dtype=torch.float32)
        labels = torch.tensor(digits.target, dtype=torch.long)

        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            train_size=split_ratio,
            random_state=seed,
            stratify=labels,
        )

        scaler = StandardScaler()
        train_features = torch.tensor(scaler.fit_transform(train_features.numpy()), dtype=torch.float32)
        test_features = torch.tensor(scaler.transform(test_features.numpy()), dtype=torch.float32)

        self.features = train_features if train else test_features
        self.labels = train_labels if train else test_labels

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class ArrayDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def load_dataset(name: Literal["digits", "synthetic", "mnist", "food101"], *, train: bool, seed: int, input_dim: int, num_classes: int, samples: int) -> Dataset:
    if name == "digits":
        return DigitsBenchmarkDataset(train=train, seed=seed)

    if name == "mnist":
        try:
            from sklearn.datasets import fetch_openml
            from sklearn.decomposition import PCA
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise RuntimeError("MNIST dataset requires scikit-learn to be installed") from exc

        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        features = torch.tensor(mnist.data, dtype=torch.float32)
        labels = torch.tensor(mnist.target.astype("int64"), dtype=torch.long)

        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            train_size=0.8,
            random_state=seed,
            stratify=labels,
        )

        scaler = StandardScaler()
        train_features = torch.tensor(scaler.fit_transform(train_features.numpy()), dtype=torch.float32)
        test_features = torch.tensor(scaler.transform(test_features.numpy()), dtype=torch.float32)

        if train_features.size(1) != input_dim:
            pca = PCA(n_components=input_dim, random_state=seed)
            train_features = torch.tensor(pca.fit_transform(train_features.numpy()), dtype=torch.float32)
            test_features = torch.tensor(pca.transform(test_features.numpy()), dtype=torch.float32)

        if train:
            target_features, target_labels = train_features, train_labels
            max_samples = min(samples, target_features.size(0))
            gen = torch.Generator().manual_seed(seed)
        else:
            target_features, target_labels = test_features, test_labels
            max_samples = min(max(256, samples // 5), target_features.size(0))
            gen = torch.Generator().manual_seed(seed + 1)

        indices = torch.randperm(target_features.size(0), generator=gen)[:max_samples]
        return ArrayDataset(target_features[indices], target_labels[indices])

    if name == "food101":
        try:
            from torchvision import datasets, transforms
        except Exception as exc:
            raise RuntimeError("Food-101 dataset requires torchvision to be installed") from exc

        side = int(input_dim ** 0.5)
        if side * side != input_dim:
            raise ValueError("food101 requires input_dim to be a perfect square")

        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((side, side)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.view(-1)),
            ]
        )
        split = "train" if train else "test"
        dataset = datasets.Food101(root="data/food101", split=split, download=True, transform=transform)

        if train:
            max_samples = min(samples, len(dataset))
            gen = torch.Generator().manual_seed(seed)
        else:
            max_samples = min(max(256, samples // 5), len(dataset))
            gen = torch.Generator().manual_seed(seed + 1)

        indices = torch.randperm(len(dataset), generator=gen)[:max_samples].tolist()
        return Subset(dataset, indices)

    if train:
        return ClusteredToyDataset(num_samples=samples, input_dim=input_dim, num_classes=num_classes, seed=seed)
    return ClusteredToyDataset(num_samples=max(256, samples // 5), input_dim=input_dim, num_classes=num_classes, seed=seed + 1)

def make_cluster_specs(num_clusters: int, input_dim: int, scale: float = 2.0) -> list[ClusterSpec]:
    centers = torch.linspace(-scale, scale, steps=num_clusters).unsqueeze(1).repeat(1, input_dim)
    return [ClusterSpec(center=centers[idx], label=idx) for idx in range(num_clusters)]
