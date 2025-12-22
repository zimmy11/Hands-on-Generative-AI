import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import re
from datetime import datetime

# --- CORE MODULES ---
from src.models.unet_model import UNet 
from src.utils.WIP_processes import Diffusion_Processes
# Assumed: Your GMM classes are in a file named gmm_utils.py or similar
from src.utils.gmm_utils import GMMDataset, GMMImageDataset 
from src.training.ldm_module import LDMLightningModule 

# --------------- SETUP --------------- #
def setup(cfg, device: torch.device):
    print(f"1. Initializing GMM Toy Setup on {device}...")

    # 1. Initialize GMM Toy Dataset
    # Define two modes for the mixture
    means = [[-3.0, -3.0], [3.0, 3.0]]
    sigmas = [0.5, 0.5]
    
    raw_gmm = GMMDataset(means, sigmas)
    # k=3 for tight coverage, sigma_pixel=1.5 for visible blobs
    toy_dataset = GMMImageDataset(
        raw_gmm, 
        img_size=cfg['image_size'], 
        sigma_pixel=1.5, # represent the standard deviation of the blob
        k=3 # number of the std. deviations for the distribution to consider in the grid
    )

    # Wrap the custom sampler in a simple Dataset class if needed, 
    # or use a simple lambda/factory for the loader.
    # For simplicity, we generate a fixed set for this toy run:
    train_data = toy_dataset.sample_as_images(1000) # Pre-generate 1000 samples
    val_data = toy_dataset.sample_as_images(200)

    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg['batch_size'], shuffle=False)

    # 2. Model Initialization
    # Since it's a toy set, we use latent_channels as the image channel (1)
    unet_model = UNet(
        in_channels=cfg['channels'], 
        out_channels=cfg['channels'], 
        features=cfg['features']
    ).to(device)

    forward_process = Diffusion_Processes(cfg)

    hparams = {
        'learning_rate': cfg['learning_rate'],
        'n_timesteps': cfg['N'],
        'batch_size': cfg['batch_size'],
        'vae_scale_factor': cfg['vae_scale_factor']
    }

    # 3. Instantiate Lightning Module
    # We pass None for VAE as we are in pixel space
    ldm_module = LDMLightningModule(
        unet_model=unet_model, 
        diffusion_process=forward_process, 
        vae_encoder=None,
        vae_decoder=None,
        hparams=hparams, 
        cfg=cfg
    )
    
    return ldm_module, train_loader, val_loader

# --------------- MAIN --------------- #
def main():
    wandb.login(key="3f785a5ef6c94fac05a13ed4a58965545976c05b")
    
    parser = argparse.ArgumentParser(description="GMM Diffusion Training")
    parser.add_argument('--config-path', type=str, default='./experiments/base_config.yaml')
    args = parser.parse_args()

    print("Loading configuarionts from base_config.yaml")
    with open(args.config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Extract parameters
    cfg = {
        'N': yaml_config.get('n_timesteps', 1000),
        'sde_type': yaml_config.get('sde_type', 'vp'),
        'epochs': yaml_config.get('epochs', 20),
        'learning_rate': yaml_config.get('learning_rate', 1e-4),
        'batch_size': yaml_config.get('batch_size', 32),
        'channels': yaml_config.get('channels', 3),
        'image_size': yaml_config.get('image_size', 32),
        'features': yaml_config.get('features', [32, 64, 128]),
        'early_stopping_patience': 15,
        'latent_channels': yaml_config.get('latent_channels', 4),
        'vae_scale_factor': yaml_config.get('vae_scale_factor', 0.18125),
        'vae_factor': yaml_config.get('vae_factor', 8),
        'validation_split_ratio': yaml_config.get('validation_split_ratio', 0.2),
        'self_attention': yaml_config.get('self_attention', False),
        'model_type': yaml_config.get('model', "LDM"),
    }
    print("Configurations loaded. cfg:")
    print(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ldm_module, train_loader, val_loader = setup(cfg, device)

    # Logging
    lr_str = str(cfg['learning_rate']).replace('.', '')
    hyper_suffix = f"T{cfg['N']}_LR{lr_str}_E{cfg['epochs']}"
    
    hyper_suffix += "_GMM"
    wandb_logger = WandbLogger(project="LDM Training", name=f"{cfg['model_type']}_{hyper_suffix}" ,config=cfg)
    
    wandb_logger.experiment.log({"config_forward": cfg})

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/',
        filename='gmm-step-{epoch:02d}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    print("Starting LDM Training...")
    trainer = Trainer(
        logger=wandb_logger,
        accelerator="auto",
        max_epochs=cfg['epochs'],
        callbacks=[checkpoint_callback],
        limit_train_batches=cfg['batch_size'], # Run full batches for GMM
        limit_val_batches=cfg['batch_size']
    )

    trainer.fit(ldm_module, train_loader, val_loader)

    #Finish
    wandb.finish()
    
    final_save_dir = os.path.join('./checkpoints', 'weights')

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        
    actual_epochs = trainer.current_epoch 
    hyper_suffix = re.sub(r'E\d+', f'E{actual_epochs}', hyper_suffix)
    hyper_suffix += f"_ts{current_timestamp}"

    final_model_filename = f"{cfg['model_type']}_final_{hyper_suffix}.pth"

    # Ensure the final save directory exists
    os.makedirs(final_save_dir, exist_ok=True)
    final_model_path = os.path.join(final_save_dir, final_model_filename)
    torch.save({'state_dict': ldm_module.state_dict()}, final_model_path)
    print(f"\n[FINAL SAVE] Final weights saved to: {final_model_path}")

if __name__ == "__main__":
    main()