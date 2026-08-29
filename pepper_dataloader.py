"""PyTorch DataLoader for the two-species, two-quality pepper dataset.

Expected directory layout::

    dataset_root/
      子弹头_好/<source_image_id>/*.png
      子弹头_差/<source_image_id>/*.png
      条子_好/<source_image_id>/*.png
      条子_差/<source_image_id>/*.png

Images originating from the same source photograph are always assigned to the
same split. This prevents near-duplicate peppers from leaking between training
and evaluation sets.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


QUALITY_TO_INDEX = {"差": 0, "好": 1}
SPECIES_TO_INDEX = {"子弹头": 0, "条子": 1}
INDEX_TO_QUALITY = {value: key for key, value in QUALITY_TO_INDEX.items()}
INDEX_TO_SPECIES = {value: key for key, value in SPECIES_TO_INDEX.items()}

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class PepperSample:
    path: Path
    quality: int
    species: int
    source_id: str

    @property
    def joint_class(self) -> tuple[int, int]:
        return self.species, self.quality

    @property
    def image_id(self) -> str:
        # The relative class/source/file form is stable and globally unique.
        return f"{self.path.parent.parent.name}/{self.source_id}/{self.path.name}"


class CompositeRGBA:
    """Flatten transparency onto a fixed RGB background."""

    def __init__(self, background_rgb: tuple[int, int, int]) -> None:
        self.background_rgb = background_rgb

    def __call__(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*self.background_rgb, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


class SquarePad:
    """Pad an RGB image to a square without changing its aspect ratio."""

    def __init__(self, fill: tuple[int, int, int]) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), self.fill)
        left = (side - image.width) // 2
        top = (side - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


class PepperDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        samples: Sequence[PepperSample],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        if not samples:
            raise ValueError("PepperDataset received an empty sample list")
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image_tensor = self.transform(image)

        return {
            "image": image_tensor.to(dtype=torch.float32),
            "quality": torch.tensor(sample.quality, dtype=torch.long),
            "species": torch.tensor(sample.species, dtype=torch.long),
            "image_id": sample.image_id,
        }


def discover_samples(dataset_root: str | Path) -> list[PepperSample]:
    """Discover and validate samples below ``dataset_root``."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples: list[PepperSample] = []
    expected_folders = {
        f"{species}_{quality}"
        for species in SPECIES_TO_INDEX
        for quality in QUALITY_TO_INDEX
    }

    for class_name in sorted(expected_folders):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        species_name, quality_name = class_name.rsplit("_", maxsplit=1)
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = path.relative_to(class_dir)
            # The first directory is the source-photo ID. If images are placed
            # directly in the class directory, the filename becomes the group.
            source_id = relative.parts[0] if len(relative.parts) > 1 else path.stem
            samples.append(
                PepperSample(
                    path=path,
                    quality=QUALITY_TO_INDEX[quality_name],
                    species=SPECIES_TO_INDEX[species_name],
                    source_id=source_id,
                )
            )

    if not samples:
        raise RuntimeError(f"No supported images found below: {root}")
    return samples


def _split_count(group_count: int, ratio: float) -> int:
    if ratio == 0 or group_count < 3:
        return 0
    return max(1, round(group_count * ratio))


def split_by_source(
    samples: Sequence[PepperSample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[PepperSample], list[PepperSample], list[PepperSample]]:
    """Stratify by joint class and keep each source photo in one split."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1")

    grouped: dict[tuple[int, int], dict[str, list[PepperSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        grouped[sample.joint_class][sample.source_id].append(sample)

    split_samples: dict[str, list[PepperSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    rng = random.Random(seed)

    for joint_class in sorted(grouped):
        source_groups = list(grouped[joint_class].values())
        rng.shuffle(source_groups)
        number_of_groups = len(source_groups)
        number_of_test_groups = _split_count(number_of_groups, test_ratio)
        number_of_val_groups = _split_count(number_of_groups, val_ratio)

        # Always retain at least one source photograph for training.
        while number_of_test_groups + number_of_val_groups >= number_of_groups:
            if number_of_test_groups >= number_of_val_groups and number_of_test_groups:
                number_of_test_groups -= 1
            elif number_of_val_groups:
                number_of_val_groups -= 1

        test_end = number_of_test_groups
        val_end = test_end + number_of_val_groups
        split_samples["test"].extend(_flatten(source_groups[:test_end]))
        split_samples["val"].extend(_flatten(source_groups[test_end:val_end]))
        split_samples["train"].extend(_flatten(source_groups[val_end:]))

    for split in split_samples.values():
        split.sort(key=lambda sample: str(sample.path))

    _validate_source_isolation(split_samples)
    return split_samples["train"], split_samples["val"], split_samples["test"]


def _flatten(groups: Iterable[Sequence[PepperSample]]) -> list[PepperSample]:
    return [sample for group in groups for sample in group]


def _validate_source_isolation(splits: dict[str, Sequence[PepperSample]]) -> None:
    seen: dict[tuple[tuple[int, int], str], str] = {}
    for split_name, samples in splits.items():
        for sample in samples:
            group_key = (sample.joint_class, sample.source_id)
            previous_split = seen.setdefault(group_key, split_name)
            if previous_split != split_name:
                raise RuntimeError(
                    f"Source group {group_key} occurs in both "
                    f"{previous_split} and {split_name}"
                )


def build_transforms(
    image_size: int = 224,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Callable[[Image.Image], Tensor], Callable[[Image.Image], Tensor]]:
    """Return training and evaluation transforms."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if len(background_rgb) != 3 or any(not 0 <= channel <= 255 for channel in background_rgb):
        raise ValueError("background_rgb must contain three integers in [0, 255]")

    common_prefix = [CompositeRGBA(background_rgb), SquarePad(background_rgb)]
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_transform = transforms.Compose(
        [
            *common_prefix,
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomAffine(
                degrees=20,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                fill=background_rgb,
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            *common_prefix,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def _balanced_sampler(samples: Sequence[PepperSample], seed: int) -> WeightedRandomSampler:
    counts = Counter(sample.joint_class for sample in samples)
    weights = [1.0 / counts[sample.joint_class] for sample in samples]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(samples),
        replacement=True,
        generator=generator,
    )


def create_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    balance_train: bool = True,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create ``train_loader``, ``val_loader`` and ``test_loader``.

    The default collator turns scalar label tensors into ``Tensor[B]`` and
    strings into ``list[str]``, matching the requested batch contract.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    samples = discover_samples(dataset_root)
    train_samples, val_samples, test_samples = split_by_source(
        samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_transform, evaluation_transform = build_transforms(
        image_size=image_size,
        background_rgb=background_rgb,
    )

    train_dataset = PepperDataset(train_samples, train_transform)
    val_dataset = PepperDataset(val_samples, evaluation_transform)
    test_dataset = PepperDataset(test_samples, evaluation_transform)

    sampler = _balanced_sampler(train_samples, seed) if balance_train else None
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader, test_loader


def describe_dataloaders(loaders: Sequence[DataLoader]) -> None:
    """Print split sizes and joint-class counts for a quick sanity check."""
    for split_name, loader in zip(("train", "val", "test"), loaders):
        dataset = loader.dataset
        if not isinstance(dataset, PepperDataset):
            continue
        counts = Counter(
            f"{INDEX_TO_SPECIES[s.species]}_{INDEX_TO_QUALITY[s.quality]}"
            for s in dataset.samples
        )
        print(f"{split_name}: {len(dataset)} images, {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    default_root = Path(__file__).parent / "辣椒单体_透明PNG" / "成品"
    data_loaders = create_dataloaders(default_root)
    describe_dataloaders(data_loaders)

    example_batch = next(iter(data_loaders[0]))
    print("image:", example_batch["image"].shape, example_batch["image"].dtype)
    print("quality:", example_batch["quality"].shape, example_batch["quality"].dtype)
    print("species:", example_batch["species"].shape, example_batch["species"].dtype)
    print("image_id type:", type(example_batch["image_id"]), example_batch["image_id"][:2])
