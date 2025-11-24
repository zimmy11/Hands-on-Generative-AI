import os
import sys
import argparse
#import yaml
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# Import all core components from your structured project modules
from src.models.unet_model import UNet  # Your custom UNet model
from src.utils.sde_utils import * 
# from src.utils.subVP_forward import ForwardProcess   # Forward process with subVP_SDE
from src.utils.vae_utils import get_vae_encoder_func # Function to load and return the VAE encoder
from src.data.base_dataset import LatentDataset       # Your custom Dataset class
from src.training.ldm_module import LDMLightningModule # Your PL module core


# --- SETUP FUNCTION ---
def setup(cfg: ForwardConfig, data_path: str, device: torch.device):
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
        full_dataset = LatentDataset(data_dir=data_path, image_size=cfg.image_size)
        
        # Define split sizes
        val_size = int(cfg.validation_split_ratio * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        # Deterministic Split for reproducibility
        torch.manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        print(f"Dataset loaded: Total {len(full_dataset)} images.")
        print(f" -> Train Loader: {len(train_dataset)} images.")

    except Exception as e:
        print(f"ERROR: Could not load data from {data_path}. Check path and dataset class. {e}")
        sys.exit(1)

    # B. Model and Diffusion Setup
    unet_model = UNet(in_channels=cfg.latent_channels, out_channels=cfg.latent_channels, features=cfg.features, ).to(device)
    vae_encoder_func = get_vae_encoder_func(device) # VAE Encoder function
    
    # Initialize ForwardProcess (contains the subVP_SDE instance)
    forward_process = DiffusionProcesses(beta_min=cfg.beta_min, beta_max=cfg.beta_max, N=cfg.N)
    sde = subVP_SDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, N=cfg.N)

    # C. Importance Sampling Calculation (IS)
    is_probabilities = None
    if cfg.use_importance_sampling:
        print("2. Calculating Importance Sampling probabilities (g(t)^2 / lambda_orig(t))...")
        # forward_process.sde_model is the subVP_SDE instance required for calculation
        is_probabilities = calculate_importance_sampling_probabilities(
            sde, 
            cfg.N, 
            device
        )

    # D. Prepare Hparams for PL Module & Early Stopping
    hparams = {
        'learning_rate': cfg.learning_rate,
        'vae_scale_factor': cfg.vae_scale_factor,
        'n_timesteps': cfg.N,
        'is_probabilities': is_probabilities, # Pass the IS tensor through hparams for access in training_step
        'batch_size': cfg.batch_size,
        'data_path': data_path
    }



    # E. Instantiate Lightning Module
    ldm_module = LDMLightningModule(
        unet_model=unet_model, 
        forward_process=forward_process, 
        vae_encoder=vae_encoder_func, 
        hparams=hparams
    )
    
    return ldm_module, train_loader, val_loader


# --- MAIN EXECUTION FUNCTION ---
def main():
    
    # 1. Argument Parsing (Used for GCP Vertex AI Custom Job configuration)
    parser = argparse.ArgumentParser(description="PyTorch Lightning LDM Training")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset directory (GCS for cloud training).')
    parser.add_argument('../../experiments', type=str, default='./base_config.yaml', help='Path to the YAML config file.')
    args = parser.parse_args()

    # Loading the configuraitons
    print(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Diffusion processes
    beta_min = yaml_config.get('beta_min', 0.1)
    beta_max = yaml_config.get('beta_max', 20.0)
    N_timesteps = yaml_config.get('n_timesteps', 1000)
    schedule = yaml_config.get("schedule", "linear")
    seed = yaml_config.get("seed", 42)

    # Forward specific parameters
    t_forward = yaml_config.get('t_forward', 1.0)
    final = yaml_config.get('final', True)
    eps = yaml_config.get('eps', 1e-5)
    closed_formula = yaml_config.get("closed_formula", True)
    
    # Reverse and Likelihood parameters
    t_0 = yaml_config.get('t_0', 1.0)
    t_1 = yaml_config.get('t_1', 0.0)
    corrector = yaml_config.get('corrector', False)
    n_corr = yaml_config.get('corrector', 50)
    target_snr = yaml_config.get('target_snr', 0.16)
    rev_type = yaml_config.get('rev_type', 'sde')

    # Training Params
    epochs = yaml_config.get('epochs', 50)
    lr = yaml_config.get('learning_rate', 0.0001)
    batch_size = yaml_config.get('batch_size', 64)
    model_type = yaml_config.get('model', 'LDM')

    # Model Params
    use_is = yaml_config.get('use_importance_sampling', True)
    latent_ch = yaml_config.get('latent_channels', 4)
    img_size = yaml_config.get('image_size', 128)
    vae_scale = yaml_config.get('vae_scale_factor', 0.18215)
    vae_factor = yaml_config.get('vae_factor', 8)
    val_split = yaml_config.get('validation_split_ratio', 0.2)
    feats = yaml_config.get('features', [128, 256, 512])
    attn = yaml_config.get('self_attention', True)
    workers = yaml_config.get('num_workers', 0)

    
    latent_h = img_size // vae_factor
    latent_w = img_size // vae_factor

    # Create the Shape Tuple (B, C, H, W)
    current_shape = (batch_size, latent_ch, latent_h, latent_w)
    
    cfg = {
        'ForwardConfig': {
            # Operational parameters (hardcoded for training logic)
            't': t_forward,
            'final': final,
            'eps': eps,
            'closed_formula': closed_formula,
            'seed': seed,
            
            # SDE Parameters from YAML
            'beta_min': beta_min,
            'beta_max': beta_max,
            'N': N_timesteps,
            'schedule': schedule,
            
            # Model/Data Parameters from YAML
            'use_importance_sampling': use_is,
            'latent_channels': latent_ch,
            'image_size': img_size,
            'vae_scale_factor': vae_scale,
            'validation_split_ratio': val_split,
            'features': feats,
            'self_attention': attn,
            'num_workers': workers,
            'data_path': args.data_path, # Overwrite YAML path with CLI arg
            
            # Training Meta
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'model': model_type
        },
        'ReverseConfig': {
            # Operational parameters
            'output_path': "reverse.pt",
            'scores': "scores.pt",
            't0': t_0,
            't1': t_1,
            'device': None,
            'dtype': None,
            'shape': current_shape,
            'seed': seed,
            'corrector': corrector,
            'n_corr': n_corr,
            'target_snr': target_snr,
            'rev_type': rev_type,

            # Shared Parameters from YAML
            'beta_min': beta_min,
            'beta_max': beta_max,
            'N': N_timesteps,
            'schedule': schedule,
            'use_importance_sampling': use_is,
            'latent_channels': latent_ch,
            'image_size': img_size,
            'vae_scale_factor': vae_scale,
            'validation_split_ratio': val_split,
            'features': feats,
            'self_attention': attn,
            'num_workers': workers,
            'data_path': args.data_path, # Overwrite YAML path with CLI arg
            
            # Meta
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'model': model_type
        },
        'LikelihoodConfig': {
            # Operational parameters
            'output_path': "reverse.pt",
            'scores': "scores.pt",
            't0': t_0,
            't1': t_1,
            'device': None,
            'dtype': None,

            # Shared Parameters from YAML
            'beta_min': beta_min,
            'beta_max': beta_max,
            'N': N_timesteps,
            'schedule': schedule,
            'use_importance_sampling': use_is,
            'latent_channels': latent_ch,
            'image_size': img_size,
            'vae_scale_factor': vae_scale,
            'validation_split_ratio': val_split,
            'features': feats,
            'self_attention': attn,
            'num_workers': workers,
            'data_path': args.data_path, # Overwrite YAML path with CLI arg
            
            # Meta
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'model': model_type
        }
    }
    
    # 3. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Initialize Modules and DataLoaders
    ldm_module, train_loader, val_loader = setup(cfg, args.data_path, device)
    
    # --- W&B and Logging Setup ---
    model_name = cfg.model
    learning_rate = cfg.learning_rate
    timesteps = cfg.N
    epochs = cfg.epochs
    self_attention = cfg.self_attention
    lr_str = str(learning_rate).replace('.', '')
    hyper_suffix = f"T{timesteps}_LR{lr_str}_E{epochs}"

    if self_attention:
        hyper_suffix += "_SA"


    # 5. W&B Initialization (Key is read automatically from WANDB_API_KEY environment variable on GCP)
    # wandb.init(
    #     project = "LDM Training",
    #     #project=os.getenv("WANDB_PROJECT", "LDM Training"), 
    #     config=cfg,
    #     name = f"{model_name}_{hyper_suffix}",
    #     reinit=True
    # )
    # The WandbLogger integrates logging with the PL Trainer
    wandb_logger = WandbLogger(project = "LDM Training",
        name=f"{model_name}_{hyper_suffix}", config=cfg,       
        #project=os.getenv("WANDB_PROJECT", "LDM Training"), 
        log_model="all")

    # 6. Checkpoint Callback (Saves model states)
    # AIP_MODEL_DIR is the standard Vertex AI output path (e.g., gs://bucket/output_ldm_pl/)
    checkpoint_path = os.getenv("AIP_MODEL_DIR", './checkpoints/')
    interim_save_dir = os.path.join(checkpoint_path, 'interim')
    os.makedirs(interim_save_dir, exist_ok=True)


    checkpoint_callback = ModelCheckpoint(
        dirpath=interim_save_dir, 
        filename='ldm-epoch{epoch:02d}-val_loss{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )


    patience = cfg.early_stopping_patience or 10

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=True,
        mode='min'
    )

    # 7. Initialize Trainer (Optimized for T4/GCP Cost Saving)
    trainer = Trainer(
        logger=wandb_logger,
        accelerator = "cuda",
        #accelerator="gpu",
        devices=1,                      # Use 1 T4 GPU
        max_epochs=cfg.epochs,
        precision=16,           # CRUCIAL: Enables Mixed Precision for speed and VRAM savings on T4
        callbacks=[checkpoint_callback, early_stopping],
        limit_train_batches=0.5, limit_val_batches=0.5 # --> we use it to test the code quickly
        # Example for quick debug run: limit_train_batches=0.1, limit_val_batches=0.1
    )

    # 8. Start Training
    print("3. Starting LDM Training...")
    trainer.fit(ldm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 9. Cleanup
    wandb.finish()

    final_save_dir = os.path.join(checkpoint_path, 'weights')

    final_model_filename = f"{model_name}_final_{hyper_suffix}.pth"

    # Ensure the final save directory exists
    os.makedirs(final_save_dir, exist_ok=True)
    final_model_path = os.path.join(final_save_dir, final_model_filename)
    torch.save({'state_dict': ldm_module.state_dict()}, final_model_path)
    print(f"\n[FINAL SAVE] Final weights saved to: {final_model_path}")


if __name__ == "__main__":
    main()
    #python -m train --data-path="C:\Users\marco\Desktop\Magistrale\ERASMUS\COURSES TUM\Practicals\Hands on Generative AI\Project\Hands-on-Generative-AI\data\train2017"