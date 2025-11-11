import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# Import all core components from your structured project modules
from src.models.unet_model import UNet  # Your custom UNet model
from src.utils.sde_utils import calculate_importance_sampling_probabilities, subVP_SDE, ForwardProcess 
# from src.utils.subVP_forward import ForwardProcess   # Forward process with subVP_SDE
from src.utils.vae_utils import get_vae_encoder_func # Function to load and return the VAE encoder
from src.data.base_dataset import LatentDataset       # Your custom Dataset class
from src.training.ldm_module import LDMLightningModule # Your PL module core


# --- GLOBAL CONFIGURATION PLACEHOLDERS (Will be overridden by config.yaml) ---
# These are just here for structural reference.
LATENT_CHANNELS = 4 
IMAGE_SIZE = 512 
FEATURES = [320, 640, 1280]
VAE_SCALE_FACTOR = 0.18215 
VALIDATION_SPLIT_RATIO = 0.1


# --- SETUP FUNCTION ---
def setup(cfg: dict, data_path: str, device: torch.device):
    """
    Sets up all model components, data loaders, and calculates the IS tensor.
    
    Args:
        cfg (dict): Configuration dictionary loaded from YAML.
        data_path (str): Path to the dataset (local path or GCS path for Dataloader).
        device (torch.device): Target device ('cuda' or 'cpu').
        
    Returns:
        tuple: (ldm_module, train_loader, val_loader)
    """
    
    print(f"1. Initializing setup on {device}...")

    # A. Data Loading & Splitting
    try:
        # Load the full dataset (assuming raw images are present in the directory)
        full_dataset = LatentDataset(data_dir=data_path, image_size=IMAGE_SIZE)
        
        # Define split sizes
        val_size = int(VALIDATION_SPLIT_RATIO * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        # Deterministic Split for reproducibility
        torch.manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

        print(f"Dataset loaded: Total {len(full_dataset)} images.")
        print(f" -> Train Loader: {len(train_dataset)} images.")

    except Exception as e:
        print(f"ERROR: Could not load data from {data_path}. Check path and dataset class. {e}")
        sys.exit(1)

    # B. Model and Diffusion Setup
    unet_model = UNet(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS, features=FEATURES).to(device)
    vae_encoder_func = get_vae_encoder_func(device) # VAE Encoder function
    
    # Initialize ForwardProcess (contains the subVP_SDE instance)
    forward_process = ForwardProcess(beta_min=cfg['beta_min'], beta_max=cfg['beta_max'], N=cfg['n_timesteps'])
    
    # C. Importance Sampling Calculation (IS)
    is_probabilities = None
    if cfg['use_importance_sampling']:
        print("2. Calculating Importance Sampling probabilities (g(t)^2 / lambda_orig(t))...")
        # forward_process.sde_model is the subVP_SDE instance required for calculation
        is_probabilities = calculate_importance_sampling_probabilities(
            forward_process.sde_model, 
            cfg['n_timesteps'], 
            device
        )

    # D. Prepare Hparams for PL Module
    hparams = {
        'learning_rate': cfg['learning_rate'],
        'vae_scale_factor': VAE_SCALE_FACTOR,
        'n_timesteps': cfg['n_timesteps'],
        'is_probabilities': is_probabilities, # Pass the IS tensor through hparams for access in training_step
        'batch_size': cfg['batch_size'],
        'data_path': data_path
    }

    # E. Instantiate Lightning Module
    ldm_module = LDMLightningModule(
        unet_model=unet_model, 
        forward_process=forward_process, 
        vae_encoder_func=vae_encoder_func, 
        hparams=hparams
    )
    
    return ldm_module, train_loader, val_loader


# --- MAIN EXECUTION FUNCTION ---
def main():
    
    # 1. Argument Parsing (Used for GCP Vertex AI Custom Job configuration)
    parser = argparse.ArgumentParser(description="PyTorch Lightning LDM Training")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset directory (GCS for cloud training).')
    parser.add_argument('--config-path', type=str, default='configs/config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # 2. Load Configuration
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 3. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Initialize Modules and DataLoaders
    ldm_module, train_loader, val_loader = setup(cfg, args.data_path, device)
    
    # --- W&B and Logging Setup ---

    # 5. W&B Initialization (Key is read automatically from WANDB_API_KEY environment variable on GCP)
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "LDM Training"), 
        config=cfg,
    )
    # The WandbLogger integrates logging with the PL Trainer
    wandb_logger = WandbLogger(project=os.getenv("WANDB_PROJECT", "LDM Training"), log_model="all")

    # 6. Checkpoint Callback (Saves model states)
    # AIP_MODEL_DIR is the standard Vertex AI output path (e.g., gs://bucket/output_ldm_pl/)
    checkpoint_path = os.getenv("AIP_MODEL_DIR", './checkpoints/')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, 
        filename='ldm-epoch{epoch:02d}-val_loss{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    # 7. Initialize Trainer (Optimized for T4/GCP Cost Saving)
    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,                      # Use 1 T4 GPU
        max_epochs=cfg['epochs'],
        precision="16-mixed",           # CRUCIAL: Enables Mixed Precision for speed and VRAM savings on T4
        callbacks=[checkpoint_callback],
        # Example for quick debug run: limit_train_batches=0.1, limit_val_batches=0.1
    )

    # 8. Start Training
    print("3. Starting LDM Training...")
    trainer.fit(ldm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 9. Cleanup
    wandb.finish()


if __name__ == "__main__":
    main()