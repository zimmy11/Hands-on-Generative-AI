import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import wandb
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
# Import all core components from your structured project modules
from src.utils.sde_utils import * 
from torchvision import transforms
from src.data.base_dataset import LatentDataset       # Your custom Dataset class
from src.training.ldm_module import LDMLightningModule # Your PL module core
import numpy as np
from torchvision.datasets import CelebA
from torchvision import transforms
#from src.models.unet_model import UNet  # Your custom UNet model
from src.models.UNet import UNet
from .vae_utils import get_vae_encoder_func
from src.models.components import EMAModel


def sample_hparams(trial):

    features_key = trial.suggest_categorical(
        "features", ["small", "medium", "large"]
    )

    FEATURE_MAP = {
        "small":  [64, 128, 256],
        "medium": [128, 256, 512],
        "large":  [256, 512, 1024]
    }
    return {
        # Optimization
        "learning_rate": trial.suggest_float("lr", 1e-5, 5e-4, log = True),
        "batch_size": trial.suggest_categorical("batch_size", [1, 2, 4, 8]),

        # Diffusion
        "sigma_min": trial.suggest_float("sigma_min", 0.01, 0.1, log=True),
        "sigma_max": trial.suggest_float("sigma_max", 20, 348),
        "n_timesteps": trial.suggest_categorical("N", [500, 1000, 2000]),

        # UNet
        "num_blocks": trial.suggest_int("num_blocks", 2, 4),
        "num_features": FEATURE_MAP[features_key],
        "embedding_dim": trial.suggest_categorical("emb_dim", [128, 256, 512]), 
        "image_size":  trial.suggest_categorical("image_size", [32, 64, 128])
    }

def make_objective(cfg):
    def objective(trial):
        wandb.init()
        h = sample_hparams(trial)

        # --- aggiorna cfg ---
        cfg.update({
            "learning_rate": h["learning_rate"],
            "batch_size": h["batch_size"],
            "sigma_min": h["sigma_min"],
            "sigma_max": h["sigma_max"],
            "N": h["n_timesteps"],
            "features": h["num_features"],
            'num_blocks': h['num_blocks']
        })

        #  training corto = proxy
        cfg["epochs"] = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ldm_module, train_loader, val_loader = setup(cfg, device = device , data_path = None)

        self_attention = cfg['self_attention']
        lr =  cfg['learning_rate']
        lr_str = str(lr).replace('.', '')
        N_timesteps = cfg['N']
        epochs = cfg['epochs']

        hyper_suffix = f"T{N_timesteps}_LR{lr_str}_E{epochs}"

        if self_attention:
            hyper_suffix += "_SA"

        wandb_logger = WandbLogger(project = "LDM Training",
        name=f"LDM_{hyper_suffix}", config=cfg)       
        #project=os.getenv("WANDB_PROJECT", "LDM Training"), 
        #log_model="all")
    
        #wandb_logger.experiment.log({"config_forward": cfg["ForwardConfig"]})

        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            max_epochs=cfg["epochs"],
            logger=wandb_logger,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_loss")
            ],
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            enable_checkpointing=False
        )
        try:
            trainer.fit(ldm_module, train_loader, val_loader)

            # metrics = {}
            # metrics["val_loss"] = trainer.callback_metrics["val_loss"].item()
            # metrics["train_loss"] = trainer.callback_metrics["train_loss"].item()
            # wandb.log(metrics)

            wandb.finish()

            return trainer.callback_metrics["val_loss"].item()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            else:
                raise e

    return objective




# --- SETUP FUNCTION ---
def setup(cfg, data_path: str, device: torch.device):
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

    forward_cfg = cfg
    
    try:

        #full_dataset = LatentDataset(data_dir=data_path, image_size=forward_cfg['image_size'])
        image_size = forward_cfg['image_size']
        transform = transforms.Compose([transforms.CenterCrop(178), transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

        full_dataset = CelebA(
            root=data_path,
            # root = data_path
            split="train",
            target_type="attr",
            transform=transform,
            download=False   
        )
        print("CelebA dataset loaded successfully.")


        # indices = [0] * 128  # Example indices for a small subset
        # full_dataset = torch.utils.data.Subset(full_dataset, indices)
        # Define split sizes
        val_size = int(forward_cfg['validation_split_ratio'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        # Deterministic Split for reproducibility
        torch.manual_seed(forward_cfg['seed'])
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        # train_dataset = full_dataset
        # val_dataset = full_dataset
        # Create DataLoaders


        train_loader = DataLoader(train_dataset, batch_size=forward_cfg['batch_size'], shuffle=True, num_workers=forward_cfg['num_workers'])# CHange Batch size
        val_loader = DataLoader(val_dataset, batch_size=forward_cfg['batch_size'], shuffle=False, num_workers=forward_cfg['num_workers']) # Change Batch size

        print(f"Dataset loaded: Total {len(full_dataset)} images.")
        print(f" -> Train Loader: {len(train_dataset)} images.")

    except Exception as e:
        print(f"ERROR: Could not load data from {data_path}. Check path and dataset class. {e}")
        sys.exit(1)

    # B. Model and Diffusion Setup
    unet_model = UNet(in_channels=forward_cfg['latent_channels']).to(device)#, out_channels=forward_cfg['latent_channels'], features=forward_cfg['features'], ).to(device)
    vae_encoder_func, vae_decoder_func = get_vae_encoder_func(device) # VAE Encoder function
    ema_model = EMAModel(unet_model).to(device)

    # Initialize ForwardProcess (contains the subVP_SDE instance)
    forward_process = Diffusion_Processes(forward_cfg)
    sde = SubVPSDE(beta_max=forward_cfg['beta_max'], beta_min=forward_cfg['beta_min'], N=forward_cfg['N'])

    # C. Importance Sampling Calculation (IS)
    is_probabilities = None
    if forward_cfg['use_importance_sampling']:
        print("2. Calculating Importance Sampling probabilities (g(t)^2 / lambda_orig(t))...")
        # forward_process.sde_model is the subVP_SDE instance required for calculation
        is_probabilities = calculate_importance_sampling_probabilities(
            sde, 
            forward_cfg['N'], 
            device
        )
    else:
        is_probabilities = torch.ones(forward_cfg['N'], device=device) / forward_cfg['N']

    # D. Prepare Hparams for PL Module & Early Stopping
    hparams = {
        'learning_rate': forward_cfg['learning_rate'],
        'vae_scale_factor': forward_cfg['vae_scale_factor'],
        'n_timesteps': forward_cfg['N'],
        'is_probabilities': is_probabilities, # Pass the IS tensor through hparams for access in training_step
        'batch_size': forward_cfg['batch_size'],
        'data_path': data_path, 
        'ema': ema_model 
    }



    # E. Instantiate Lightning Module
    ldm_module = LDMLightningModule(
        unet_model=unet_model, 
        forward_process=forward_process, 
        vae_encoder=vae_encoder_func, 
        vae_decoder=vae_decoder_func,
        hparams=hparams, 
        cfg = cfg
    )
    
    return ldm_module, train_loader, val_loader
