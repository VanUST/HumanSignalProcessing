import torch
import lightning as L
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from utils import load_config
from model_definition import WaveletCNN, WaveletClassifier
from dataset_wrapper import USS_40khz_datamodule
from augmentations import get_augmentations

def get_hyperparameter_search_space():
    return {
        'model.num_classes': 5,
        'model.wavelet_type': tune.choice(['haar', 'db1', 'sym2']),
        'model.wavelet_level': tune.choice([2, 3, 4]),
        'model.conv_channels': tune.choice([[8]]),
        'model.kernel_sizes': tune.choice([[3]]),
        'model.num_conv_layers': tune.choice([[1]]),
        'model.token_len': 1,
        'training.learning_rate': tune.loguniform(1e-5, 1e-2),
        'training.batch_size': tune.choice([16, 32, 64]),
        'training.num_workers': 4,
        'training.optimizer': tune.choice(['adamw', 'adam']),
        'training.weight_decay': tune.uniform(0.0, 0.1),
        'training.amsgrad': tune.choice([True, False]),
        'data.data_dir': 'path/to/your/csv/files',
        'data.selected_numbers': [1,2,3],
        'data.num_points_to_read': 2000,
        'data.val_split': 0.2,
        'data.normalize': True,
        'data.augmentations': False
    }

def train_tune(config):
    # Set random seed
    L.seed_everything(42)

    # Create model
    model_params = {
        'num_classes': config['model.num_classes'],
        'wavelet_type': config['model.wavelet_type'],
        'wavelet_level': config['model.wavelet_level'],
        'conv_channels': config['model.conv_channels'],
        'kernel_sizes': config['model.kernel_sizes'],
        'num_conv_layers': config['model.num_conv_layers'],
        'token_len': config['model.token_len']
    }
    wavelet_cnn = WaveletCNN(**model_params)

    # Wrap model with LightningModule
    classifier = WaveletClassifier(
        model=wavelet_cnn,
        learning_rate=config['training.learning_rate'],
        optimizer=config['training.optimizer'],
        weight_decay=config['training.weight_decay'],
        amsgrad=config['training.amsgrad']
    )

    # Create augmentations if enabled
    augmentations = get_augmentations() if config['data.augmentations'] else None

    # Instantiate DataModule
    data_module = USS_40khz_datamodule(
        data_dir=config['data.data_dir'],
        selected_numbers=config['data.selected_numbers'],
        batch_size=config['training.batch_size'],
        num_points_to_read=config['data.num_points_to_read'],
        num_workers=config['training.num_workers'],
        normalize=config['data.normalize'],
        val_split=config['data.val_split'],
        augmentations=augmentations
    )

    # Define trainer with TuneReportCallback
    metrics = {'val_loss': 'val_loss', 'val_acc': 'val_acc'}
    trainer = L.Trainer(
        max_epochs=-1,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=False,
        enable_progress_bar=False,
        callbacks=[TuneReportCallback(metrics, on='validation_end')]
    )

    # Train the model
    trainer.fit(classifier, datamodule=data_module)

def main_tune():
    # Load base configuration
    base_config = load_config()

    search_space = get_hyperparameter_search_space()

    analysis = tune.run(
        tune.with_parameters(train_tune, num_epochs=base_config['training']['num_epochs']),
        resources_per_trial=base_config['ray_tune']['resources_per_trial'],
        metric='val_acc',
        mode='max',
        config=search_space,
        num_samples=base_config['ray_tune']['num_samples']
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == '__main__':
    main_tune()
