import sys
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import wandb

# Import project modules
from src.utils.sde_utils import Diffusion_Processes, SubVPSDE, VPSDE, VESDE, calculate_importance_sampling_probabilities
from src.training.ldm_module import LDMLightningModule
from src.models.UNet import UNet
from .vae_utils import get_vae_encoder_func
from src.models.components import EMAModel


# --- VISUALIZATION FUNCTION ---
def log_denoising_step_wandb(x_tensor, step, total_steps, caption_prefix="Denoising"):
    """
    Logs intermediate denoising steps to WandB.

    Args:
        x_tensor (torch.Tensor): Current batch of images (B, C, H, W)
        step (int): Current step (counting down from N to 0)
        total_steps (int): Total number of steps
        caption_prefix (str): Prefix for caption
    """
    wandb.init()
    # Normalize images from [-1, 1] -> [0, 1] for visualization
    x_vis = x_tensor.detach().cpu().clamp(-1.0, 1.0)
    x_vis = (x_vis + 1.0) / 2.0

    # Create a grid of first 16 samples
    n_show = min(16, x_vis.shape[0])
    grid = torchvision.utils.make_grid(x_vis[:n_show], nrow=4)

    # Compute percentage complete
    percent_complete = 100 * (1 - step / total_steps)
    caption = f"{caption_prefix} - Step {step}/{total_steps} ({percent_complete:.1f}%)"

    # Log image to WandB
    wandb.log({
        "generated_samples/process": wandb.Image(grid, caption=caption)
    })


# --- SETUP FUNCTION ---
def setup(cfg, data_path: str, device: torch.device):
    """
    Sets up model, data loaders, SDE process, EMA, and optional importance sampling.

    Args:
        cfg (dict): Configuration dictionary from YAML
        data_path (str): Path to dataset
        device (torch.device): Target device

    Returns:
        tuple: (ldm_module, train_loader, val_loader, test_loader)
    """
    print(f"1. Initializing setup on {device}...")

    dataset_type = cfg.get('dataset_type', 'Celeb')
    image_size = cfg['image_size']

    # ---------------------------
    # A. Data Loading & Splitting
    # ---------------------------
    try:
        if dataset_type == "Celeb":
            print("Loading CelebA dataset...")

            transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

            full_dataset = datasets.CelebA(
                root=data_path,
                split="train",
                target_type="attr",
                transform=transform,
                download=False
            )

            # Split into train/val/test
            val_size = int(cfg['validation_split_ratio'] * len(full_dataset))
            test_size = val_size
            train_size = len(full_dataset) - val_size - test_size

            torch.manual_seed(cfg['seed'])
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )

            train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
            val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
            test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

            print(f"Dataset loaded: Total {len(full_dataset)} images.")
            print(f" -> Train Loader: {len(train_dataset)} images.")

        else:
            print("Loading MNIST dataset...")

            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            train_full = datasets.MNIST(root=data_path, train=True, transform=transform, download=False)
            test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=False)

            val_ratio = cfg['validation_split_ratio']
            val_size = int(val_ratio * len(train_full))
            train_size = len(train_full) - val_size

            generator = torch.Generator().manual_seed(cfg['seed'])
            train_dataset, val_dataset = random_split(train_full, [train_size, val_size], generator=generator)

            train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
            val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
            test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

            print(f"Dataset loaded: Total {len(train_full)} images.")
            print(f" -> Train Loader: {len(train_dataset)} images.")

    except Exception as e:
        print(f"ERROR: Could not load data from {data_path}. Check path and dataset class. {e}")
        sys.exit(1)

    # ---------------------------
    # B. Model & Diffusion Setup
    # ---------------------------
    unet_model = UNet(
        in_channels=cfg['latent_channels'],
        out_channels=cfg['latent_channels'],
        num_attributes=cfg['num_attributes']
    ).to(device)

    vae_encoder_func, vae_decoder_func = get_vae_encoder_func(device)
    ema_model = EMAModel(unet_model).to(device)
    forward_process = Diffusion_Processes(cfg)

    if cfg['sde_type'] == 'subVP':
        sde = SubVPSDE(beta_max=cfg['beta_max'], beta_min=cfg['beta_min'], N=cfg['N'])
    elif cfg['sde_type'] == 'vp':
        sde = VPSDE(beta_max=cfg['beta_max'], beta_min=cfg['beta_min'], N=cfg['N'])
    else:
        sde = VESDE(sigma_min=cfg['sigma_min'], sigma_max=cfg['sigma_max'], N=cfg['N'])

    # ---------------------------
    # C. Importance Sampling Probabilities
    # ---------------------------
    if cfg['use_importance_sampling']:
        print("2. Calculating Importance Sampling probabilities...")
        is_probabilities = calculate_importance_sampling_probabilities(sde, cfg['N'], device)
    else:
        is_probabilities = torch.ones(cfg['N'], device=device) / cfg['N']

    print("Likelihood weighting:", cfg.get('likelihood_weighting', True))

    # ---------------------------
    # D. Prepare hparams for Lightning Module
    # ---------------------------
    hparams = {
        'learning_rate': cfg['learning_rate'],
        'vae_scale_factor': cfg['vae_scale_factor'],
        'n_timesteps': cfg['N'],
        'is_probabilities': is_probabilities,
        'batch_size': cfg['batch_size'],
        'data_path': data_path,
        'ema': ema_model,
        'likelihood_weighting': cfg.get('likelihood_weighting', True),
        'dataset_type': dataset_type
    }

    # ---------------------------
    # E. Instantiate Lightning Module
    # ---------------------------
    ldm_module = LDMLightningModule(
        unet_model=unet_model,
        forward_process=forward_process,
        vae_encoder=vae_encoder_func,
        vae_decoder=vae_decoder_func,
        hparams=hparams,
        cfg=cfg
    )

    return ldm_module, train_loader, val_loader, test_loader
