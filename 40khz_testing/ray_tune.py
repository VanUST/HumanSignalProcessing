import torch
import lightning as L
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils import load_config
from model_definition import WaveletCNN, WaveletClassifier
from dataset_wrapper import Datamodule_40khz
from augmentations import get_augmentations

def get_hyperparameter_search_space():
    return {
        'model.num_classes': 7,
        'model.wavelet_type': tune.choice(['haar', 'db1', 'sym2']),
        'model.dropout': tune.uniform(0,0.7),
        'model.wavelet_level': tune.choice([2, 3, 4]),
        'model.conv_channels': tune.choice([8, 16, 32, 64]),
        'model.kernel_size': tune.choice([3, 5, 7,9]),
        'model.num_conv_layers': tune.choice([1, 2,3,4]),
        'model.token_len': 1,
        'training.learning_rate': tune.loguniform(1e-5, 1e-2),
        'training.batch_size': tune.choice([7,2,4]),
        'training.num_workers': 4,
        'training.optimizer': tune.choice(['adamw']),
        'training.weight_decay': tune.uniform(0.0, 0.1),
        'training.amsgrad': tune.choice([True, False]),
        'data.augmentations': tune.choice([True, False]),
        'data.folder_names': tune.choice([
                                    ['/home/giga_ivan/projects/HandyRobotics/40khz_testing/test_data_40khz/DAN']
                                  , 
                                    ['/home/giga_ivan/projects/HandyRobotics/40khz_testing/test_data_40khz/VAN']
                                  ]),  # Hardcoded in config as per your instruction
        'data.mode': 'full_channels',  # or 'mode2', depending on your dataset
        'data.val_split': 0.25,
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
        'kernel_size': config['model.kernel_size'],
        'num_conv_layers': config['model.num_conv_layers'],
        'token_len': config['model.token_len'],
        'dropout': config['model.dropout']
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
    data_module = Datamodule_40khz(
        folder_names=config['data.folder_names'],
        batch_size=config['training.batch_size'],
        num_workers=config['training.num_workers'],
        train_val_split=config['data.val_split'],
        mode=config['data.mode'],
        transform=augmentations
    )

    # Define trainer with TuneReportCallback
    metrics = {'val_loss': 'val_loss', 'val_acc': 'val_acc'}
    trainer = L.Trainer(
        max_epochs=-1,  # You can adjust the number of epochs as needed
        logger=False,
        enable_progress_bar=False,
        callbacks=[TuneReportCallback(metrics, on='validation_end'),
                   EarlyStopping(
                                monitor='val_loss',
                                patience=50,
                                verbose=True,
                                mode='min'  
                                )
                    ]
    )

    # Train the model
    trainer.fit(classifier, datamodule=data_module)

def main_tune():
    # Load base configuration
    base_config = load_config('/home/giga_ivan/projects/HandyRobotics/40khz_testing/config.json')

    search_space = get_hyperparameter_search_space()

    analysis = tune.run(
        tune.with_parameters(train_tune),
        max_concurrent_trials=1,
        resources_per_trial=base_config['ray_tune']['resources_per_trial'],
        metric='val_loss',
        mode='min',
        config=search_space,
        num_samples=base_config['ray_tune']['num_samples']
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == '__main__':
    main_tune()
