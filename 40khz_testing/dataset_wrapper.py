import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
import torch.nn.functional as F


class Dataset_40khz(Dataset):
    def __init__(self, folder_names, mode='single_channel', transform=None):

        self.folder_names = folder_names
        self.mode = mode
        self.transform = transform
        self.samples = []
        self.labels = []
        self._build_samples()
        self._build_label_mapping()
    
    def _build_samples(self):
        if self.mode == 'single_channel':
            # Each sample is a single file
            for folder_name in self.folder_names:
                folder_path = os.path.join(folder_name)
                files = glob.glob(os.path.join(folder_path, '*.txt'))
                for file in files:
                    basename = os.path.basename(file)
                    # Extract 'jest' from filename
                    jest = basename.split('_')[-1].split('.txt')[0]
                    self.samples.append(file)
                    self.labels.append(jest)
        elif self.mode == 'full_channels':
            # Group files with same 'jest' and 'try'
            sample_dict = {}
            for folder_name in self.folder_names:
                folder_path = os.path.join(folder_name)
                files = glob.glob(os.path.join(folder_path, '*.txt'))
                for file in files:
                    basename = os.path.basename(file)
                    parts = basename.split('_')
                    try_num = parts[2]
                    jest = parts[-1].split('.txt')[0]
                    key = (folder_name, jest, try_num)
                    if key not in sample_dict:
                        sample_dict[key] = []
                    sample_dict[key].append(file)
            for key, file_list in sample_dict.items():
                self.samples.append(file_list)
                self.labels.append(key[1])  # jest
        else:
            raise ValueError("Mode must be 'single_channel' or 'full_channels'")
    
    def _build_label_mapping(self):
        # Create a mapping from label strings to integers
        unique_labels = sorted(set(self.labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.integer_encoded_labels = [self.label_to_index[label] for label in self.labels]
        self.num_classes = len(self.label_to_index)

    def __len__(self):
        return len(self.samples)
    
    def _load_data(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        # Remove '$' and ';', then convert to float
        data = [float(line.strip().replace('$', '').replace(';', '')) for line in lines if line.strip()]
        return np.array(data,dtype=np.float32)
    
    def __getitem__(self, idx):
        if self.mode == 'single_channel':
            file = self.samples[idx]
            data = self._load_data(file)
            integer_label = self.integer_encoded_labels[idx]
            label = F.one_hot(torch.tensor(integer_label), num_classes=self.num_classes)
            if self.transform:
                data = self.transform(data,sample_rate = 40000)
            return torch.tensor(data, dtype=torch.float32), label.to(dtype=torch.float32)
        elif self.mode == 'full_channels':
            files = self.samples[idx]
            data_list = [self._load_data(file) for file in files]
            data = np.stack(data_list, axis=0)
            integer_label = self.integer_encoded_labels[idx]
            label = F.one_hot(torch.tensor(integer_label), num_classes=self.num_classes)
            if self.transform:
                data = self.transform(data,sample_rate = 40000)
            return torch.tensor(data, dtype=torch.float32), label.to(dtype=torch.float32)

class Datamodule_40khz(L.LightningDataModule):

    def __init__(self, folder_names, train_val_split=0.8, batch_size=32, num_workers=0, mode='full_channels', transform=None):
        super().__init__()
        self.folder_names = folder_names
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.transform = transform
        full_dataset = Dataset_40khz(self.folder_names, mode=self.mode, transform=self.transform)
        # Get labels
        labels = full_dataset.integer_encoded_labels
        # Perform stratified split
        train_indices, val_indices = self._stratified_split(labels, self.train_val_split)
        # Create subsets
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        

    def _stratified_split(self, labels, train_ratio):
        from collections import defaultdict
        import random
        # Group indices by class label
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        # Split indices for each class
        train_indices = []
        val_indices = []
        for label, indices in label_to_indices.items():
            random.shuffle(indices)
            split = int(train_ratio * len(indices))
            train_indices.extend(indices[:split])
            val_indices.extend(indices[split:])
        # Shuffle the overall indices
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        return train_indices, val_indices

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)