import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.DelayedEarlyStopping import DelayedEarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import re
from datetime import datetime
# Import all core components from your structured project modules
from src.models.unet_model import UNet  # Your custom UNet model
from src.utils.sde_utils import * 
# from src.utils.subVP_forward import ForwardProcess   # Forward process with subVP_SDE
from src.utils.vae_utils import get_vae_encoder_func # Function to load and return the VAE encoder
from src.data.base_dataset import LatentDataset       # Your custom Dataset class
from src.training.ldm_module import LDMLightningModule # Your PL module core
from google.cloud import storage
import numpy as np
from torchvision.datasets import CelebA
from torchvision import datasets, transforms


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

    forward_cfg = cfg['ForwardConfig']
    
    try:

        # Load the full dataset (assuming raw images are present in the directory)
        # if data_path.startswith("gs://"):
        #     local_data_path = "/tmp/train_data"
        #     download_gcs_folder(data_path, local_data_path)
        #     data_path = local_data_path


        #full_dataset = LatentDataset(data_dir=data_path, image_size=forward_cfg['image_size'])
        image_size = forward_cfg['image_size']
        transform = transforms.Compose([transforms.CenterCrop(178), transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

        full_dataset = datasets.CelebA(
            root="./data",
            split="train",
            target_type="attr",
            transform=transform,
            download=False   
        )

        indices = [0]  # Example indices for a small subset
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        # Define split sizes
        # val_size = int(forward_cfg['validation_split_ratio'] * len(full_dataset))
        # train_size = len(full_dataset) - val_size
        
        # Deterministic Split for reproducibility
        # torch.manual_seed(forward_cfg['seed'])
        # train_dataset, val_dataset = random_split(
        #     full_dataset, [train_size, val_size]
        # )
        train_dataset = full_dataset
        val_dataset = full_dataset
        # Create DataLoaders


        train_loader = DataLoader(train_dataset, batch_size=forward_cfg['batch_size'], shuffle=True, num_workers=forward_cfg['num_workers'])# CHange Batch size
        val_loader = DataLoader(val_dataset, batch_size=forward_cfg['batch_size'], shuffle=True, num_workers=forward_cfg['num_workers']) # Change Batch size

        print(f"Dataset loaded: Total {len(full_dataset)} images.")
        print(f" -> Train Loader: {len(train_dataset)} images.")

    except Exception as e:
        print(f"ERROR: Could not load data from {data_path}. Check path and dataset class. {e}")
        sys.exit(1)

    # B. Model and Diffusion Setup
    unet_model = UNet(in_channels=forward_cfg['latent_channels'], out_channels=forward_cfg['latent_channels'], features=forward_cfg['features'], ).to(device)
    vae_encoder_func, vae_decoder_func = get_vae_encoder_func(device) # VAE Encoder function
    
    # Initialize ForwardProcess (contains the subVP_SDE instance)
    forward_process = DiffusionProcesses(cfg)
    sde = VESDE()

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
        'data_path': data_path
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

# --- GCS Utility Function --- 
# Downloads a folder from GCS to local path
def download_gcs_folder(bucket_path, local_path):
    client = storage.Client()
    bucket_name, prefix = bucket_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    os.makedirs(local_path, exist_ok=True)
    for blob in blobs:
        dest_path = os.path.join(local_path, os.path.relpath(blob.name, prefix))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)


def download_gcs_file(gcs_path: str, local_dir: str) -> str:
    """
    Scarica un singolo file da GCS e ritorna il path locale.
    gcs_path: gs://bucket/.../file.ext
    local_dir: directory locale dove salvare
    """
    client = storage.Client()
    path = gcs_path.replace("gs://", "")
    bucket_name, blob_path = path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    os.makedirs(local_dir, exist_ok=True)
    local_filename = os.path.join(local_dir, os.path.basename(blob_path))
    blob.download_to_filename(local_filename)
    return local_filename



def extract_epoch_from_filename(ckpt_filename: str) -> int:
    """
    Estrae l'ultima epoch da un filename tipo:
    'LDM_final_T1000_LR00001_E24_SA_ts20251125_144930.pth'
    """
    match = re.search(r'_E(\d+)', ckpt_filename)
    if match:
        return int(match.group(1))
    else:
        return 0



# --- MAIN EXECUTION FUNCTION ---
def main():

    wandb.login(key="3f785a5ef6c94fac05a13ed4a58965545976c05b")
    # 1. Argument Parsing (Used for GCP Vertex AI Custom Job configuration)
    parser = argparse.ArgumentParser(description="PyTorch Lightning LDM Training")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset directory (GCS for cloud training).')
    parser.add_argument('--config-path', type=str, default='./experiments/base_config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                    help='Path to checkpoint file to resume from. Accepts local path or gs://bucket/path/file.(ckpt|pth|pt)')
    
    # Override parameters from command line if provided
    parser.add_argument('--beta-min', type=float)
    parser.add_argument('--beta-max', type=float)
    parser.add_argument('--n-timesteps', type=int)
    parser.add_argument('--schedule', type=str)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--t-forward', type=float)
    parser.add_argument('--final', type=bool)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--closed-formula', type=bool)

    parser.add_argument('--t0', type=float)
    parser.add_argument('--t1', type=float)
    parser.add_argument('--corrector', type=bool)
    parser.add_argument('--n-corr', type=int)
    parser.add_argument('--target-snr', type=float)
    parser.add_argument('--rev-type', type=str)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--model', type=str)

    parser.add_argument('--use-importance-sampling', type=bool)
    parser.add_argument('--latent-channels', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--vae-scale-factor', type=float)
    parser.add_argument('--vae-factor', type=int)
    parser.add_argument('--validation-split', type=float)
    parser.add_argument('--features', nargs='+', type=int)
    parser.add_argument('--self-attention', type=bool)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--early-stopping-patience', type=int)

    args = parser.parse_args()

    # ---------------------------
    # 2. Load YAML config
    # ---------------------------    
    
    # Loading the configuraitons
    torch.set_float32_matmul_precision('medium')
    
    print(f"Loading configuration from: {args.config_path}")
    config_path = args.config_path
    if config_path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_path = config_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        yaml_bytes = blob.download_as_bytes()
        yaml_config = yaml.safe_load(yaml_bytes)
    else:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

    resume_ckpt_cli = args.resume_checkpoint
    resume_ckpt_local = None
    _downloaded_tmp_ckpt = None

    if resume_ckpt_cli:
        print(f"[INFO] resume-checkpoint provided: {resume_ckpt_cli}")
        # se è un path GCS, lo scarichiamo in /tmp
        if resume_ckpt_cli.startswith("gs://"):
            try:
                tmp_dir = "/tmp/checkpoints"
                print(f"[INFO] downloading checkpoint from GCS to {tmp_dir} ...")
                resume_ckpt_local = download_gcs_file(resume_ckpt_cli, tmp_dir)
                _downloaded_tmp_ckpt = resume_ckpt_local
                print(f"[INFO] checkpoint downloaded to {resume_ckpt_local}")
            except Exception as e:
                print(f"[ERROR] could not download checkpoint from {resume_ckpt_cli}: {e}")
                sys.exit(1)
        else:
            # local path: usalo direttamente (su Vertex AI questo sarà tipicamente un percorso già disponibile)
            if not os.path.exists(resume_ckpt_cli):
                print(f"[ERROR] resume checkpoint not found at local path: {resume_ckpt_cli}")
                sys.exit(1)
            resume_ckpt_local = resume_ckpt_cli


    def get_param(key, cli_value):
        return cli_value if cli_value is not None else yaml_config.get(key)

    beta_min   = get_param('beta_min', args.beta_min)
    beta_max   = get_param('beta_max', args.beta_max)
    N_timesteps = get_param('n_timesteps', args.n_timesteps)
    schedule   = get_param('schedule', args.schedule)
    seed       = get_param('seed', args.seed)

    t_forward  = get_param('t_forward', args.t_forward)
    final      = get_param('final', args.final)
    eps        = get_param('eps', args.eps)
    closed_formula = get_param('closed_formula', args.closed_formula)

    t_0        = get_param('t_0', args.t0)
    t_1        = get_param('t_1', args.t1)
    corrector  = get_param('corrector', args.corrector)
    n_corr     = get_param('n_corr', args.n_corr)
    target_snr = get_param('target_snr', args.target_snr)
    rev_type   = get_param('rev_type', args.rev_type)

    epochs     = get_param('epochs', args.epochs)
    lr         = get_param('learning_rate', args.learning_rate)
    batch_size = get_param('batch_size', args.batch_size)
    model_type = get_param('model', args.model)

    use_is     = get_param('use_importance_sampling', args.use_importance_sampling)
    latent_ch  = get_param('latent_channels', args.latent_channels)
    image_size   = get_param('image_size', args.image_size)
    vae_scale  = get_param('vae_scale_factor', args.vae_scale_factor)
    vae_factor = get_param('vae_factor', args.vae_factor)
    val_split  = get_param('validation_split_ratio', args.validation_split)
    feats      = get_param('features', args.features)
    attn       = get_param('self_attention', args.self_attention)
    workers    = get_param('num_workers', args.num_workers)
    early_stopping_patience    = get_param('early_stopping_patience', args.early_stopping_patience)
    
    latent_h = image_size // vae_factor
    latent_w = image_size // vae_factor

    # Create the Shape Tuple (B, C, H, W)
    current_shape = (batch_size, latent_ch, latent_h, latent_w)
    # 3. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
            'image_size': image_size,
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
            'model': model_type, 
            'early_stopping_patience': early_stopping_patience
        },
        'ReverseConfig': {
            # Operational parameters
            'output_path': "reverse.pt",
            'scores': "scores.pt",
            't0': t_0,
            't1': t_1,
            'device': device,
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
            'image_size': image_size,
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
            'model': model_type, 
            'early_stopping_patience': early_stopping_patience
        },
        'LikelihoodConfig': {
            # Operational parameters
            'output_path': "reverse.pt",
            'scores': "scores.pt",
            't0': t_0,
            't1': t_1,
            'device': device,
            'dtype': None,

            # Shared Parameters from YAML
            'beta_min': beta_min,
            'beta_max': beta_max,
            'N': N_timesteps,
            'schedule': schedule,
            'use_importance_sampling': use_is,
            'latent_channels': latent_ch,
            'image_size': image_size,
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
            'model': model_type, 
            'early_stopping_patience': early_stopping_patience
        }
    }
    
    

    # 4. Initialize Modules and DataLoaders
    ldm_module, train_loader, val_loader = setup(cfg, args.data_path, device)
    
    # --- W&B and Logging Setup ---
    self_attention = cfg['ForwardConfig']['self_attention']
    lr_str = str(lr).replace('.', '')
    hyper_suffix = f"T{N_timesteps}_LR{lr_str}_E{epochs}"

    if self_attention:
        hyper_suffix += "_SA"
    if resume_ckpt_cli:
        hyper_suffix += "_RESUME"



    # The WandbLogger integrates logging with the PL Trainer
    wandb_logger = WandbLogger(project = "LDM Training",
        name=f"{model_type}_{hyper_suffix}", config=cfg)       
        #project=os.getenv("WANDB_PROJECT", "LDM Training"), 
        #log_model="all")
    
    wandb_logger.experiment.log({"config_forward": cfg["ForwardConfig"]})

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
        every_n_epochs=50,
        save_top_k=1, 
        save_last=False
        #save_on_train_epoch_end=True
    )


    patience = cfg['ForwardConfig']['early_stopping_patience'] or 15

    # early_stopping = DelayedEarlyStopping(
    #     start_epoch=50, 
    #     monitor='val_loss',
    #     patience=patience,
    #     mode='min',
    #     verbose=True
    # )



    # 7. Start Training
    print("Starting LDM Training...")

    ckpt_path_for_trainer = None

    if resume_ckpt_local:
        last_epoch_done = extract_epoch_from_filename(os.path.basename(resume_ckpt_local))

        print(f"[INFO] Preparing to resume from checkpoint: {resume_ckpt_local}")
        # Proviamo a caricare il file e vedere se contiene 'state_dict' (tipico .pth salvati manualmente)
        try:
            ckpt_dict = torch.load(resume_ckpt_local, map_location="cpu")
            # Se è un dict con 'state_dict' (tipico salvataggio manuale {'state_dict': model.state_dict()})
            if isinstance(ckpt_dict, dict) and 'state_dict' in ckpt_dict:
                print("[INFO] Found 'state_dict' in checkpoint — loading into LightningModule...")
                ldm_module.load_state_dict(ckpt_dict['state_dict'], strict=False)
                # non passiamo ckpt_path al trainer (già caricato nello stato del model)
                ckpt_path_for_trainer = None
            else:
                # Potrebbe essere un full Lightning checkpoint (.ckpt) o uno state_dict puro
                # Se sembra un Lightning checkpoint (contiene 'pytorch-lightning_version' o 'optimizer_states'), usalo come ckpt_path
                if isinstance(ckpt_dict, dict) and ('pytorch-lightning_version' in ckpt_dict or 'optimizer_states' in ckpt_dict):
                    print("[INFO] Detected Lightning checkpoint — will resume via trainer.ckpt_path")
                    ckpt_path_for_trainer = resume_ckpt_local
                else:
                    # Altrimenti, assumiamo sia uno state_dict puro (mapping nome->tensor)
                    try:
                        ldm_module.load_state_dict(ckpt_dict, strict=False)
                        print("[INFO] Loaded checkpoint as model state_dict.")
                        ckpt_path_for_trainer = None
                    except Exception as e:
                        print(f"[WARNING] Could not auto-load checkpoint into model: {e}")
                        # fallback: pass path to trainer (Trainer proverà a gestire .ckpt)
                        ckpt_path_for_trainer = resume_ckpt_local
        except Exception as e:
            # se torch.load fallisce, proviamo a passare il path direttamente al trainer
            print(f"[WARNING] torch.load failed on {resume_ckpt_local}: {e}")
            ckpt_path_for_trainer = resume_ckpt_local
        total_epochs = args.epochs if args.epochs else cfg['ForwardConfig']['epochs']
        remaining_epochs = max(total_epochs - last_epoch_done, 0)
        print(f"[INFO] Trainer will run {remaining_epochs} additional epochs.")

    else: 
        remaining_epochs = args.epochs if args.epochs else cfg['ForwardConfig']['epochs']
   

    # 8. Initialize Trainer (Optimized for T4/GCP Cost Saving)
    trainer = Trainer(
        logger=wandb_logger,
        accelerator="cpu",
        devices="auto",                      # Use 1 T4 GPU
        max_epochs=epochs,
        # precision="16-mixed",           # CRUCIAL: Enables Mixed Precision for speed and VRAM savings on T4
        callbacks=[checkpoint_callback], #, early_stopping],
        limit_train_batches=1, 
        limit_val_batches=1 # --> we use it to test the code quickly
    )

    trainer.fit(ldm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 9. Cleanup
    wandb.finish()

    final_save_dir = os.path.join(checkpoint_path, 'weights')


    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        
    actual_epochs = trainer.current_epoch 
    hyper_suffix = re.sub(r'E\d+', f'E{actual_epochs}', hyper_suffix)
    hyper_suffix += f"_ts{current_timestamp}"

    final_model_filename = f"{model_type}_final_{hyper_suffix}.pth"

    # Ensure the final save directory exists
    os.makedirs(final_save_dir, exist_ok=True)
    final_model_path = os.path.join(final_save_dir, final_model_filename)
    torch.save({'state_dict': ldm_module.state_dict()}, final_model_path)
    print(f"\n[FINAL SAVE] Final weights saved to: {final_model_path}")

    if _downloaded_tmp_ckpt and os.path.exists(_downloaded_tmp_ckpt):
        try:
            os.remove(_downloaded_tmp_ckpt)
            print(f"[INFO] removed temporary checkpoint file {_downloaded_tmp_ckpt}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
    #python -m train --data-path="C:\Users\marco\Desktop\Magistrale\ERASMUS\COURSES TUM\Practicals\Hands on Generative AI\Project\Hands-on-Generative-AI\data\train2017" --resume-checkpoint="C:\Users\marco\Desktop\Magistrale\ERASMUS\COURSES TUM\Practicals\Hands on Generative AI\Project\Hands-on-Generative-AI\checkpoints\weights\LDM_final_T1000_LR00001_E24_SA_ts20251125_144930.pth" --epochs=30