import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary
import lightning as L

from utils import load_config
from model_definition import WaveletCNN, WaveletClassifier
from dataset_wrapper import Datamodule_40khz
from augmentations import get_augmentations

def main():
    # Load configuration
    config = load_config('/home/giga_ivan/projects/HandyRobotics/40khz_testing/config.json')

    # Set random seed for reproducibility
    L.seed_everything(42)

    # Create model
    wavelet_cnn = WaveletCNN(**config['model'])

    # Wrap the model with LightningModule
    classifier = WaveletClassifier(
        model=wavelet_cnn,
        learning_rate=config['training']['learning_rate'],
        optimizer=config['training']['optimizer'],
        weight_decay=config['training']['weight_decay'],
        amsgrad=config['training']['amsgrad']
    )

    # Create augmentations
    augmentations = get_augmentations() if config['data']['augmentations'] else None

    # Instantiate the DataModule
    data_module = Datamodule_40khz(
        folder_names=config['data']['folder_names'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        train_val_split=config['data']['val_split'],
        mode = config['data']['mode'],
        transform=augmentations
    )
    
    train_dataloader,val_dataloader = data_module.train_dataloader(), data_module.val_dataloader()

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='wavelet-cnn-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )

    model_summary_callback  = ModelSummary(max_depth=-1)

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=10,
        verbose=True,
        mode='max'
    )

    # Instantiate the Logger
    logger = TensorBoardLogger(config['training']['tb_dir'], name="")

    # Instantiate the Trainer
    trainer = Trainer(
        max_epochs=-1,
        accelerator='auto',
        strategy= 'auto',
        devices='auto',
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler='simple',
        callbacks= [model_summary_callback]
    )

    # Train the model
    trainer.fit(classifier, val_dataloaders=val_dataloader,train_dataloaders=train_dataloader)

    # # Test the model
    # trainer.test(classifier, val_dataloaders=val_dataloader,train_dataloaders=train_dataloader)

if __name__ == '__main__':
    main()
