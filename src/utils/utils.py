import sys
import wandb
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# Project-specific modules
from src.utils.sde_utils import * 
from torchvision import transforms
from src.data.base_dataset import LatentDataset       # Custom dataset class
from src.training.ldm_module import LDMLightningModule # Lightning module
from src.models.UNet import UNet
from .vae_utils import get_vae_encoder_func
from src.models.components import EMAModel
from torchvision.datasets import CelebA
import numpy as np



# -------------------------------------
# 1. Setup function (model + dataloaders)
# -------------------------------------
def setup(cfg, device: torch.device, data_path: str):
    """
    Prepares all model components, dataloaders, and importance sampling tensor.
    
    Args:
        cfg (dict): Configuration dictionary.
        device (torch.device): Target device ('cuda' or 'cpu').
        data_path (str): Path to dataset (optional, for local or cloud).
        
    Returns:
        tuple: (ldm_module, train_loader, val_loader)
    """
    print(f"1. Initializing setup on {device}...")

    # -------------------------
    # A. Data Loading & Split
    # -------------------------
    try:
        image_size = cfg['image_size']

        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # Load CelebA dataset
        full_dataset = CelebA(
            root="../data",  # Change to data_path if needed
            split="train",
            target_type="attr",
            transform=transform,
            download=False
        )

        # Split into train/validation sets
        val_size = int(cfg['validation_split_ratio'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        torch.manual_seed(cfg['seed'])
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

        print(f"Dataset loaded: Total {len(full_dataset)} images.")
        print(f" -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    except Exception as e:
        print(f"ERROR: Could not load dataset from {data_path}. {e}")
        sys.exit(1)

    # -------------------------
    # B. Model & Diffusion Setup
    # -------------------------
    unet_model = UNet(in_channels=cfg['latent_channels']).to(device)
    vae_encoder_func, vae_decoder_func = get_vae_encoder_func(device)
    ema_model = EMAModel(unet_model).to(device)

    # Initialize diffusion process and SDE
    forward_process = Diffusion_Processes(cfg)
    sde = SubVPSDE(beta_max=cfg['beta_max'], beta_min=cfg['beta_min'], N=cfg['N'])

    # -------------------------
    # C. Importance Sampling (IS)
    # -------------------------
    if cfg.get('use_importance_sampling', False):
        print("2. Calculating importance sampling probabilities...")
        is_probabilities = calculate_importance_sampling_probabilities(sde, cfg['N'], device)
    else:
        is_probabilities = torch.ones(cfg['N'], device=device) / cfg['N']

    # -------------------------
    # D. Prepare hparams for Lightning Module
    # -------------------------
    hparams = {
        'learning_rate': cfg['learning_rate'],
        'vae_scale_factor': cfg['vae_scale_factor'],
        'n_timesteps': cfg['N'],
        'is_probabilities': is_probabilities,
        'batch_size': cfg['batch_size'],
        'data_path': data_path,
        'ema': ema_model
    }

    # -------------------------
    # E. Instantiate Lightning Module
    # -------------------------
    ldm_module = LDMLightningModule(
        unet_model=unet_model,
        forward_process=forward_process,
        vae_encoder=vae_encoder_func,
        vae_decoder=vae_decoder_func,
        hparams=hparams,
        cfg=cfg
    )

    return ldm_module, train_loader, val_loader
