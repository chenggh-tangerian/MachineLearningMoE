from __future__ import annotations

import math
import importlib
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Literal
from pathlib import Path

import torch
from torch.utils.data import Dataset


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

        self.task = "classification"
        self.num_classes = num_classes

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
        self.task = "classification"
        self.num_classes = 10

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class ArrayDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        task: str = "classification",
        num_classes: int | None = None,
        vocab_size: int | None = None,
        sequence_length: int | None = None,
    ):
        self.features = features
        self.labels = labels
        self.task = task
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class DatasetSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], **metadata: Any):
        self.dataset = dataset
        self.indices = indices
        for key, value in metadata.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


_WIKITEXT2_VOCAB_CACHE: dict[tuple[int, int], dict[str, int]] = {}

_WIKITEXT2_URLS = [
    "https://research.metamind.io/wikitext/wikitext-2-v1.zip",
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
]


def _pick_indices(length: int, count: int, seed: int) -> list[int]:
    count = min(length, count)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:count].tolist()


def _square_side(input_dim: int) -> int:
    side = int(math.isqrt(input_dim))
    if side * side != input_dim:
        raise ValueError("image datasets require input_dim to be a perfect square")
    return side


def _build_cifar100_transform(input_dim: int):
    try:
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("CIFAR-100 dataset requires torchvision to be installed") from exc

    side = _square_side(input_dim)
    return transforms.Compose(
        [
            transforms.Resize((side, side)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor.view(-1)),
        ]
    )


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _download_wikitext2(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "wikitext-2-v1.zip"
    extracted_dir = root / "wikitext-2"

    if not extracted_dir.exists():
        if not archive_path.exists():
            last_error: Exception | None = None
            for url in _WIKITEXT2_URLS:
                try:
                    with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
                        handle.write(response.read())
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise RuntimeError("Unable to download WikiText2") from last_error
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)

    base_dir = root / "wikitext-2"
    return {
        "train": base_dir / "wiki.train.tokens",
        "valid": base_dir / "wiki.valid.tokens",
        "test": base_dir / "wiki.test.tokens",
    }


def _read_wikitext2_split(split: str, root: Path) -> list[str]:
    try:
        datasets = importlib.import_module("datasets")
        hf_split = {"train": "train", "valid": "validation", "test": "test"}[split]
        dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=hf_split)
        return [row["text"] for row in dataset if row["text"].strip()]
    except Exception:
        paths = _download_wikitext2(root)
        file_path = paths[split]
        return file_path.read_text(encoding="utf-8").splitlines()


def _load_wikitext2_vocab(seed: int, sequence_length: int):
    cache_key = (seed, sequence_length)
    if cache_key in _WIKITEXT2_VOCAB_CACHE:
        return _WIKITEXT2_VOCAB_CACHE[cache_key]

    train_lines = _read_wikitext2_split("train", Path("data/wikitext2"))
    token_set: set[str] = set()
    for line in train_lines:
        token_set.update(_tokenize_text(line))
    tokens = ["<unk>", "<eos>"] + sorted(token_set)
    vocab = {token: index for index, token in enumerate(tokens)}
    _WIKITEXT2_VOCAB_CACHE[cache_key] = vocab
    return vocab


class WikiText2SequenceDataset(Dataset):
    def __init__(self, train: bool, seed: int, samples: int, sequence_length: int):
        self.sequence_length = sequence_length
        self.task = "language_modeling"

        vocab = _load_wikitext2_vocab(seed, sequence_length)
        split = "train" if train else "test"
        lines = _read_wikitext2_split(split, Path("data/wikitext2"))

        token_ids: list[int] = []
        eos_index = vocab["<eos>"]
        for line in lines:
            tokens = _tokenize_text(line)
            if tokens:
                token_ids.extend(vocab[token] for token in tokens if token in vocab)
                token_ids.append(eos_index)

        if len(token_ids) <= sequence_length:
            raise RuntimeError("WikiText2 split is too small for the requested sequence length")

        tensor_ids = torch.tensor(token_ids, dtype=torch.long)
        block_size = sequence_length + 1
        usable_tokens = (tensor_ids.numel() // block_size) * block_size
        tensor_ids = tensor_ids[:usable_tokens]
        chunks = tensor_ids.view(-1, block_size)

        inputs = chunks[:, :-1]
        targets = chunks[:, 1:]

        if train:
            max_sequences = min(samples, inputs.size(0))
            indices = _pick_indices(inputs.size(0), max_sequences, seed)
        else:
            max_sequences = min(max(256, samples // 5), inputs.size(0))
            indices = _pick_indices(inputs.size(0), max_sequences, seed + 1)

        self.inputs = inputs[indices]
        self.targets = targets[indices]
        self.vocab_size = len(vocab)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


def load_dataset(
    name: Literal["digits", "synthetic", "mnist", "food101", "cifar100", "wikitext2"],
    *,
    train: bool,
    seed: int,
    input_dim: int,
    num_classes: int,
    samples: int,
    sequence_length: int = 64,
) -> Dataset:
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
        return ArrayDataset(target_features[indices], target_labels[indices], task="classification", num_classes=10)

    if name == "cifar100":
        try:
            from torchvision import datasets
        except Exception as exc:
            raise RuntimeError("CIFAR-100 dataset requires torchvision to be installed") from exc

        transform = _build_cifar100_transform(input_dim)
        dataset = datasets.CIFAR100(root="data/cifar100", train=train, download=True, transform=transform)

        if train:
            max_samples = min(samples, len(dataset))
            indices = _pick_indices(len(dataset), max_samples, seed)
        else:
            max_samples = min(max(256, samples // 5), len(dataset))
            indices = _pick_indices(len(dataset), max_samples, seed + 1)

        return DatasetSubset(dataset, indices, task="classification", num_classes=100)

    if name == "wikitext2":
        return WikiText2SequenceDataset(train=train, seed=seed, samples=samples, sequence_length=sequence_length)

    if name == "food101":
        try:
            from torchvision import datasets, transforms
        except Exception as exc:
            raise RuntimeError("Food-101 dataset requires torchvision to be installed") from exc

        side = _square_side(input_dim)

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
            indices = _pick_indices(len(dataset), max_samples, seed)
        else:
            max_samples = min(max(256, samples // 5), len(dataset))
            indices = _pick_indices(len(dataset), max_samples, seed + 1)

        return DatasetSubset(dataset, indices, task="classification", num_classes=101)

    if train:
        return ClusteredToyDataset(num_samples=samples, input_dim=input_dim, num_classes=num_classes, seed=seed)
    return ClusteredToyDataset(num_samples=max(256, samples // 5), input_dim=input_dim, num_classes=num_classes, seed=seed + 1)


def make_cluster_specs(num_clusters: int, input_dim: int, scale: float = 2.0) -> list[ClusterSpec]:
    centers = torch.linspace(-scale, scale, steps=num_clusters).unsqueeze(1).repeat(1, input_dim)
    return [ClusterSpec(center=centers[idx], label=idx) for idx in range(num_clusters)]
