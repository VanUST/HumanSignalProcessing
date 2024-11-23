import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import lightning as L
import numpy as np
from audiomentations import Compose
from preprocess_utils import parse_and_preprocess_csv  # Assuming you have a module for parsing CSV files

class USS_40khz_dataset(Dataset):
    def __init__(self, directory, selected_numbers, num_points_to_read=2000,
                 normalize=True, augmentations=None):
        """
        Initializes the dataset by parsing and preprocessing the CSV files,
        and organizing them based on the selected numbers.

        Args:
            directory (str): Path to the directory containing CSV files.
            selected_numbers (list of int): List of numbers to include in the dataset.
            num_points_to_read (int): Maximum number of points to read from each file.
            normalize (bool): Whether to normalize the time series data.
            augmentations (Compose): Audiomentations Compose object with augmentations.
        """
        self.selected_numbers = selected_numbers
        self.num_points_to_read = num_points_to_read
        self.normalize = normalize
        self.augmentations = augmentations

        # Parse and preprocess the data
        self.grouped_data = self.parse_and_preprocess_csv(directory, num_points_to_read)

        # Filter the data based on selected_numbers
        self.selected_data = {}
        for number in selected_numbers:
            if number in self.grouped_data:
                for class_id, arrays in self.grouped_data[number].items():
                    if class_id not in self.selected_data:
                        self.selected_data[class_id] = []
                    self.selected_data[class_id].extend(arrays)
            else:
                print(f"Number {number} not found in the data. Skipping.")

        if not self.selected_data:
            raise ValueError("No data found for the selected numbers.")

        # Create a sorted list of unique class_ids
        self.class_ids = sorted(self.selected_data.keys())
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        self.num_classes = len(self.class_ids)

        # Create a list of tuples (timeseries, class_idx)
        self.data = []
        for class_id, arrays in self.selected_data.items():
            class_idx = self.class_to_idx[class_id]
            for array in arrays:
                self.data.append((array, class_idx))

        print(f"Dataset initialized with {len(self.data)} samples across {self.num_classes} classes.")

    def parse_and_preprocess_csv(self, directory, num_points_to_read):
        # Implement your CSV parsing logic here
        # For the purpose of this example, I'll assume you have a function in preprocess_utils
        return parse_and_preprocess_csv(directory, num_points_to_read)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        timeseries, class_idx = self.data[idx]
        timeseries = np.array(timeseries,dtype=np.float32)

        if self.normalize:
            mean = timeseries.mean()
            std = timeseries.std()
            if std > 0:
                timeseries = (timeseries - mean) / std

        # Apply augmentations if any
        if self.augmentations:
            timeseries = self.augmentations(samples=timeseries,sample_rate = 40000)
            timeseries = torch.tensor(timeseries, dtype=torch.float)

        # One-hot encode the label
        class_idx = torch.tensor(class_idx)
        one_hot_label = F.one_hot(class_idx, num_classes=self.num_classes).float()

        return one_hot_label, timeseries


class USS_40khz_datamodule(L.LightningDataModule):
    def __init__(self, data_dir: str, selected_numbers: list, batch_size: int = 32,
                 num_points_to_read: int = 2000, num_workers: int = 4, normalize: bool = True,
                 val_split: float = 0.2, augmentations=None):
        """
        PyTorch Lightning DataModule for USS_40khz_dataset.

        Args:
            data_dir (str): Path to the directory containing CSV files.
            selected_numbers (list): List of numbers to include in the dataset.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.
            num_points_to_read (int, optional): Max points to read from each file. Defaults to 2000.
            num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
            normalize (bool, optional): Whether to normalize the time series data. Defaults to True.
            val_split (float, optional): Proportion of data to use for validation. Defaults to 0.2.
            augmentations (Compose, optional): Audiomentations Compose object. Defaults to None.
        """
        super().__init__()
        self.data_dir = data_dir
        self.selected_numbers = selected_numbers
        self.batch_size = batch_size
        self.num_points_to_read = num_points_to_read
        self.num_workers = num_workers
        self.normalize = normalize
        self.val_split = val_split
        self.augmentations = augmentations

        full_dataset = USS_40khz_dataset(
            directory=self.data_dir,
            selected_numbers=self.selected_numbers,
            num_points_to_read=self.num_points_to_read,
            normalize=self.normalize,
            augmentations=self.augmentations
        )

        total_length = len(full_dataset)
        val_length = int(total_length * self.val_split)
        train_length = total_length - val_length

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        print(f"Data splits: Train={train_length}, Val={val_length}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
