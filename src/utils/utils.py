import sys
import optuna
import wandb
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import wandb
# Import all core components from your structured project modules
from src.utils.sde_utils import * 
from torchvision import transforms
from src.training.ldm_module import LDMLightningModule # Your PL module core
from torchvision.datasets import CelebA
from torchvision import transforms
#from src.models.unet_model import UNet  # Your custom UNet model
from src.models.UNet import UNet
from .vae_utils import get_vae_encoder_func
from src.models.components import EMAModel


# --- VISUALIZATION FUNCTION ---
def log_denoising_step_wandb(x_tensor, step, total_steps, caption_prefix="Denoising"):
    """
    Callback per loggare gli step intermedi del reverse process su WandB.
    
    Args:
        x_tensor: Il batch di immagini correnti (B, C, H, W)
        step: Lo step corrente (int) (contando alla rovescia da N a 0)
        total_steps: Il numero totale di step
    """
    # 1. Denormalizza/Clampa le immagini per visualizzazione
    # Assumiamo che il modello lavori in [-1, 1], portiamo a [0, 1]
    wandb.init()
    x_vis = x_tensor.detach().cpu().clamp(-1.0, 1.0)
    x_vis = (x_vis + 1.0) / 2.0
    
    # 2. Crea una grid (es. primi 16 sample)
    n_show = min(16, x_vis.shape[0])
    grid = torchvision.utils.make_grid(x_vis[:n_show], nrow=4)
    
    # 3. Logga su WandB
    # Usiamo lo 'step' di WandB globale se disponibile, altrimenti logghiamo come media panel
    # In genere per il denoising process si preferisce loggare un'immagine con caption
    percent_complete = 100 * (1 - step / total_steps)
    caption = f"{caption_prefix} - Step {step}/{total_steps} ({percent_complete:.1f}%)"
    
    wandb.log({
        f"generated_samples/process": wandb.Image(grid, caption=caption)
    })


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
        #test_size = val_size
        train_size = len(full_dataset) - val_size #- test_size

        # Deterministic Split for reproducibility
        torch.manual_seed(forward_cfg['seed'])
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        test_dataset = val_dataset 
        # train_dataset = full_dataset
        # val_dataset = full_dataset
        # Create DataLoaders


        train_loader = DataLoader(train_dataset, batch_size=forward_cfg['batch_size'], shuffle=True, num_workers=forward_cfg['num_workers'])# CHange Batch size
        val_loader = DataLoader(val_dataset, batch_size=forward_cfg['batch_size'], shuffle=False, num_workers=forward_cfg['num_workers']) # Change Batch size
        test_loader = DataLoader(test_dataset, batch_size=forward_cfg['batch_size'], shuffle=False, num_workers=forward_cfg['num_workers']) # Change Batch size


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


    print("Likelihood weighting:", forward_cfg.get('likelihood_weighting', True))
    # D. Prepare Hparams for PL Module & Early Stopping
    hparams = {
        'learning_rate': forward_cfg['learning_rate'],
        'vae_scale_factor': forward_cfg['vae_scale_factor'],
        'n_timesteps': forward_cfg['N'],
        'is_probabilities': is_probabilities, # Pass the IS tensor through hparams for access in training_step
        'batch_size': forward_cfg['batch_size'],
        'data_path': data_path, 
        'ema': ema_model , 
        'likelihood_weighting': forward_cfg.get('likelihood_weighting', True)
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
    
    return ldm_module, train_loader, val_loader, test_loader
