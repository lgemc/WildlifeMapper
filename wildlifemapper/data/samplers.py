"""
Advanced sampling strategies for handling class imbalance in wildlife detection datasets.
Inspired by HerdNet's sampling approaches.
"""

import torch
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import Iterator, List, Optional, Dict
from collections import Counter
import random


class ClassAwareBatchSampler(Sampler):
    """
    Samples elements to ensure balanced representation of classes in each batch.
    Particularly effective for datasets with significant class imbalance.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 8,
        samples_per_class: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Args:
            dataset: Dataset with class labels accessible via targets or annotations
            batch_size: Total batch size
            samples_per_class: Number of samples per class in each batch. If None, distributes evenly
            shuffle: Whether to shuffle samples within each class
            drop_last: Whether to drop the last incomplete batch
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Extract class information from dataset
        self._build_class_indices()

        # Determine samples per class
        if samples_per_class is None:
            self.samples_per_class = max(1, batch_size // len(self.class_indices))
        else:
            self.samples_per_class = samples_per_class

        self.total_samples_per_batch = self.samples_per_class * len(self.class_indices)

        if self.total_samples_per_batch > batch_size:
            print(f"Warning: samples_per_class * num_classes ({self.total_samples_per_batch}) > batch_size ({batch_size})")

    def _build_class_indices(self):
        """Build mapping of class labels to sample indices."""
        self.class_indices = {}

        for idx in range(len(self.dataset)):
            try:
                # Try different ways to access labels depending on dataset structure
                if hasattr(self.dataset, 'get_labels'):
                    labels = self.dataset.get_labels(idx)
                elif hasattr(self.dataset, 'targets'):
                    labels = self.dataset.targets[idx]
                else:
                    # For COCO-style datasets, get labels from annotations
                    sample = self.dataset[idx]
                    if isinstance(sample, dict) and 'target' in sample:
                        target = sample['target']
                        if isinstance(target, dict) and 'labels' in target:
                            labels = target['labels']
                        else:
                            labels = target
                    else:
                        # Last resort: assume sample is (image, target) tuple
                        _, target = sample
                        if isinstance(target, dict) and 'labels' in target:
                            labels = target['labels']
                        else:
                            labels = target

                # Handle different label formats
                if torch.is_tensor(labels):
                    labels = labels.tolist()
                elif not isinstance(labels, list):
                    labels = [labels]

                # Add index to each class
                for label in labels:
                    if isinstance(label, (int, float)):
                        label = int(label)
                        if label not in self.class_indices:
                            self.class_indices[label] = []
                        self.class_indices[label].append(idx)

            except Exception as e:
                print(f"Warning: Could not extract labels for sample {idx}: {e}")
                # Default to class 0 if we can't extract labels
                if 0 not in self.class_indices:
                    self.class_indices[0] = []
                self.class_indices[0].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with balanced class representation."""
        # Shuffle indices within each class if requested
        if self.shuffle:
            for class_label in self.class_indices:
                random.shuffle(self.class_indices[class_label])

        # Calculate number of batches
        min_samples_per_class = min(len(indices) for indices in self.class_indices.values())
        max_batches = min_samples_per_class // self.samples_per_class

        if max_batches == 0:
            return iter([])

        # Generate batches
        for batch_idx in range(max_batches):
            batch = []

            # Sample from each class
            for class_label, indices in self.class_indices.items():
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch.extend(indices[start_idx:end_idx])

            # Shuffle the batch
            if self.shuffle:
                random.shuffle(batch)

            # Trim to batch size if necessary
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]

            # Only yield if we have enough samples or not dropping last
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        min_samples_per_class = min(len(indices) for indices in self.class_indices.values())
        max_batches = min_samples_per_class // self.samples_per_class
        return max_batches


class WeightedClassSampler(WeightedRandomSampler):
    """
    Weighted random sampler that automatically computes class weights based on frequency.
    Gives higher probability to minority classes.
    """

    def __init__(
        self,
        dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True
    ):
        """
        Args:
            dataset: Dataset with class labels
            num_samples: Number of samples to draw. If None, uses len(dataset)
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset

        # Compute class frequencies and weights
        class_counts = self._compute_class_counts()
        weights = self._compute_sample_weights(class_counts)

        if num_samples is None:
            num_samples = len(dataset)

        super().__init__(weights, num_samples, replacement)

    def _compute_class_counts(self) -> Dict[int, int]:
        """Compute frequency of each class in the dataset."""
        class_counts = Counter()

        for idx in range(len(self.dataset)):
            try:
                # Extract labels similar to ClassAwareBatchSampler
                sample = self.dataset[idx]
                if isinstance(sample, dict) and 'target' in sample:
                    target = sample['target']
                    if isinstance(target, dict) and 'labels' in target:
                        labels = target['labels']
                    else:
                        labels = target
                else:
                    _, target = sample
                    if isinstance(target, dict) and 'labels' in target:
                        labels = target['labels']
                    else:
                        labels = target

                # Handle different label formats
                if torch.is_tensor(labels):
                    labels = labels.tolist()
                elif not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    if isinstance(label, (int, float)):
                        class_counts[int(label)] += 1

            except Exception:
                # Default to class 0 if we can't extract labels
                class_counts[0] += 1

        return dict(class_counts)

    def _compute_sample_weights(self, class_counts: Dict[int, int]) -> List[float]:
        """Compute inverse frequency weights for each sample."""
        # Compute inverse frequencies
        total_samples = sum(class_counts.values())
        class_weights = {
            class_label: total_samples / count
            for class_label, count in class_counts.items()
        }

        # Assign weight to each sample
        sample_weights = []

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                if isinstance(sample, dict) and 'target' in sample:
                    target = sample['target']
                    if isinstance(target, dict) and 'labels' in target:
                        labels = target['labels']
                    else:
                        labels = target
                else:
                    _, target = sample
                    if isinstance(target, dict) and 'labels' in target:
                        labels = target['labels']
                    else:
                        labels = target

                # Handle different label formats
                if torch.is_tensor(labels):
                    labels = labels.tolist()
                elif not isinstance(labels, list):
                    labels = [labels]

                # Use weight of the first label (can be modified for multi-label)
                if labels and isinstance(labels[0], (int, float)):
                    weight = class_weights.get(int(labels[0]), 1.0)
                else:
                    weight = 1.0

                sample_weights.append(weight)

            except Exception:
                sample_weights.append(1.0)

        return sample_weights


class MinorityOversamplingBatchSampler(Sampler):
    """
    Batch sampler that oversamples minority classes with different augmentations.
    Particularly effective for datasets with extreme class imbalance.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 8,
        minority_boost_factor: float = 2.0,
        min_class_samples: int = 2,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: Dataset with class labels
            batch_size: Target batch size
            minority_boost_factor: Factor by which to boost minority classes
            min_class_samples: Minimum samples per class in each batch
            shuffle: Whether to shuffle samples
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.minority_boost_factor = minority_boost_factor
        self.min_class_samples = min_class_samples
        self.shuffle = shuffle

        # Build class indices and compute sampling probabilities
        self._build_class_indices()
        self._compute_sampling_probabilities()

    def _build_class_indices(self):
        """Build mapping of class labels to sample indices."""
        self.class_indices = {}

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                if isinstance(sample, dict) and 'target' in sample:
                    target = sample['target']
                    if isinstance(target, dict) and 'labels' in target:
                        labels = target['labels']
                    else:
                        labels = target
                else:
                    _, target = sample
                    if isinstance(target, dict) and 'labels' in target:
                        labels = target['labels']
                    else:
                        labels = target

                if torch.is_tensor(labels):
                    labels = labels.tolist()
                elif not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    if isinstance(label, (int, float)):
                        label = int(label)
                        if label not in self.class_indices:
                            self.class_indices[label] = []
                        self.class_indices[label].append(idx)

            except Exception:
                if 0 not in self.class_indices:
                    self.class_indices[0] = []
                self.class_indices[0].append(idx)

    def _compute_sampling_probabilities(self):
        """Compute sampling probabilities with minority class boosting."""
        class_counts = {k: len(v) for k, v in self.class_indices.items()}
        max_count = max(class_counts.values())

        # Compute boost factors (inverse frequency with boost)
        self.class_boost = {}
        for class_label, count in class_counts.items():
            boost = (max_count / count) * self.minority_boost_factor
            self.class_boost[class_label] = min(boost, 10.0)  # Cap boost to prevent extreme values

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with minority class oversampling."""
        # Create expanded pool with boosted minorities
        expanded_indices = []

        for class_label, indices in self.class_indices.items():
            boost_factor = int(self.class_boost[class_label])
            expanded_indices.extend(indices * boost_factor)

        if self.shuffle:
            random.shuffle(expanded_indices)

        # Generate batches
        for i in range(0, len(expanded_indices), self.batch_size):
            batch = expanded_indices[i:i + self.batch_size]

            if len(batch) == self.batch_size:
                yield batch
            elif len(batch) >= self.batch_size // 2:  # Include partial batches if reasonably sized
                yield batch

    def __len__(self) -> int:
        """Return approximate number of batches."""
        total_expanded = sum(
            len(indices) * int(self.class_boost[class_label])
            for class_label, indices in self.class_indices.items()
        )
        return total_expanded // self.batch_size