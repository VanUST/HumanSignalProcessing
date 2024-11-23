import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import ptwt
import torchmetrics
import lightning as L

class WaveletCNN(nn.Module):
    def __init__(self, num_classes, wavelet_type='haar', wavelet_level=2,
                 conv_channels=[8], kernel_sizes=[3], num_conv_layers=1, token_len=1):
        """
        Initializes the WaveletCNN model with configurable wavelet and convolutional parameters.

        Args:
            num_classes (int): Number of output classes for classification.
            wavelet_type (str): Type of wavelet to use (e.g., 'haar', 'db1', 'sym2'). Default is 'haar'.
            wavelet_level (int): Level of wavelet decomposition. Default is 2.
            conv_channels (list of int): List specifying the number of output channels
                                         for each convolutional layer in the pipelines.
                                         Length should match num_conv_layers.
            kernel_sizes (list of int): List specifying the kernel size for each
                                        convolutional layer. Length should match num_conv_layers.
            num_conv_layers (int): Number of convolutional layers in each convolutional pipeline.
            token_len (int): Length of the token after adaptive pooling.
        """
        super(WaveletCNN, self).__init__()
        self.num_classes = num_classes
        self.level = wavelet_level
        self.wavelet_type = wavelet_type
        self.num_conv_layers = num_conv_layers
        self.token_len = token_len


        # Create convolutional pipelines for each wavelet coefficient tensor
        # Number of pipelines = 1 (cA_J) + J (cD_j for j=1 to J)
        self.num_pipelines = 1 + self.level  # cA_J and cD_1 to cD_J
        self.conv_pipelines = nn.ModuleList()
        for _ in range(self.num_pipelines):
            layers = []
            in_channels = 1  # Assuming input has 1 channel
            for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.LeakyReLU())
                layers.append(nn.MaxPool1d(kernel_size=2))
                in_channels = out_channels  # Update for next layer
            self.conv_pipelines.append(nn.Sequential(*layers))

        # Adaptive pooling to ensure fixed output size regardless of input size
        self.global_pool = nn.AdaptiveAvgPool1d(self.token_len)  # Outputs (batch_size, channels, token_len)

        # Calculate the total number of features after concatenation
        # Each pipeline outputs conv_channels[-1] features after pooling
        total_features = conv_channels[-1] * self.num_pipelines * self.token_len

        # Final fully connected layer
        self.fc = nn.Linear(total_features, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, init_size).

        Returns:
            torch.Tensor: Logits for each class of shape (batch_size, num_classes).
        """
        # Ensure input is even-length as required by ptwt
        if x.size(-1) % (2 ** self.level) != 0:
            new_size = (x.size(-1) // (2 ** self.level)) * (2 ** self.level)
            x = x[:, :, :new_size]

        # Apply Wavelet Transform using ptwt
        # wavedec returns a list: [cA_J, cD_J, cD_{J-1}, ..., cD_1]
        coefficients = ptwt.wavedec(x, self.wavelet_type, mode='zero', level=self.level)

        # Initialize list to hold all coefficients
        all_coefficients = []

        # Process approximation and detail coefficients
        for idx, coeff in enumerate(coefficients):
            # coeff is of shape (batch_size,1, length)
            # Pass through the corresponding convolutional pipeline
            out = self.conv_pipelines[idx](coeff)
            out = self.global_pool(out)  # (batch_size, channels, token_len)
            all_coefficients.append(out)

        # Concatenate along the channel dimension
        concatenated = torch.cat(all_coefficients, dim=1)  # (batch_size, total_channels, token_len)

        # Flatten the tensor for the fully connected layer
        concatenated = concatenated.view(concatenated.size(0), -1)  # (batch_size, total_features)

        # Final classification layer
        logits = self.fc(concatenated)  # (batch_size, num_classes)

        return logits

    def crop_tensor(self, tensor, expected_size):
        """
        Crops the tensor along the last dimension to the expected size.
        If the tensor is larger, it is cropped. If smaller, it is padded with zeros.

        Args:
            tensor (torch.Tensor): Input tensor of shape (batch_size, channels, length).
            expected_size (int): Desired size of the last dimension.

        Returns:
            torch.Tensor: Tensor cropped or padded to the expected size.
        """
        current_size = tensor.size(-1)
        if current_size > expected_size:
            tensor = tensor[:, :, :expected_size]
        elif current_size < expected_size:
            padding = expected_size - current_size
            tensor = F.pad(tensor, (0, padding))
        return tensor

class WaveletClassifier(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adamw',
                 weight_decay: float = 0.0,
                 amsgrad: bool = False):
        """
        PyTorch Lightning Module for WaveletCNN classification.

        Args:
            model (nn.Module): The WaveletCNN model.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            optimizer (str, optional): Optimizer type ('adamw' or 'adam'). Defaults to 'adamw'.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.0.
            amsgrad (bool, optional): Whether to use the AMSGrad variant of the optimizer. Defaults to False.
        """
        super(WaveletClassifier, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=self.model.num_classes,average='macro')
        self.train_precision = torchmetrics.Precision(task='multiclass',num_classes=self.model.num_classes, average='macro')
        self.train_recall = torchmetrics.Recall(task='multiclass',num_classes=self.model.num_classes, average='macro')

        self.val_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=self.model.num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task='multiclass',num_classes=self.model.num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task='multiclass',num_classes=self.model.num_classes, average='macro')

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, signal_length).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (tuple): Tuple containing (one_hot_labels, timeseries).
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        labels, timeseries = batch
        logits = self(timeseries.unsqueeze(1))

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        labels_id = torch.argmax(labels, dim=1)

        # Metrics
        self.train_accuracy(preds, labels_id )
        self.train_precision(preds, labels_id )
        self.train_recall(preds, labels_id )

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (tuple): Tuple containing (one_hot_labels, timeseries).
            batch_idx (int): Batch index.
        """
        labels, timeseries = batch
        logits = self(timeseries.unsqueeze(1))

        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        labels_id = torch.argmax(labels, dim=1)

        # Metrics
        self.val_accuracy(preds, labels_id)
        self.val_precision(preds, labels_id)
        self.val_recall(preds, labels_id)

        # Log metrics
        self.log('val_loss', loss,prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True,prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True,prog_bar=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True,prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if self.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad
            )
        elif self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer
