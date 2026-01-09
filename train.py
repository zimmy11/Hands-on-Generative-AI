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
import re
from datetime import datetime
# Import all core components from your structured project modules
from src.utils.sde_utils import * 
# from src.utils.subVP_forward import ForwardProcess   # Forward process with subVP_SDE
from src.utils.utils import setup



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

    #wandb.login(key="3f785a5ef6c94fac05a13ed4a58965545976c05b")
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
    parser.add_argument('--sde-type', type=str)

    parser.add_argument('--guidance-scale', type=bool)
    parser.add_argument('--conditional', type=int)
    parser.add_argument('--num_attributes', type=int)
    parser.add_argument('--cfg_mask_prob', type=str)

    args = parser.parse_args()

    # ---------------------------
    # 2. Load YAML config
    # ---------------------------    
    
    # Loading the configuraitons
    torch.set_float32_matmul_precision('medium')
    
    print(f"Loading configuration from: {args.config_path}")
    config_path = args.config_path
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    resume_ckpt_cli = args.resume_checkpoint
    resume_ckpt_local = None
    _downloaded_tmp_ckpt = None

    if resume_ckpt_cli:
        print(f"[INFO] resume-checkpoint provided: {resume_ckpt_cli}")

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
    sde_type    = get_param('sde_type', args.sde_type)
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
    likelihood_weighting = get_param('likelihood_weighting', False)
    guidance_scale = get_param('guidance_scale', args.guidance_scale)
    conditional   = get_param('conditional', args.conditional)
    num_attributes = get_param('num_attributes', args.num_attributes)
    cfg_mask_prob  = get_param('cfg_mask_prob', args.cfg_mask_prob)
    
    latent_h = image_size // vae_factor
    latent_w = image_size // vae_factor

    # Create the Shape Tuple (B, C, H, W)
    current_shape = (batch_size, latent_ch, latent_h, latent_w)
    # 3. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path

    cfg = {
        'N': N_timesteps,
        'sde_type': sde_type,
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'model_type': model_type,
        'latent_channels': latent_ch,
        'image_size': image_size,
        'vae_scale_factor': vae_scale,
        'vae_factor': vae_factor,
        'validation_split_ratio': val_split,
        'features': feats,
        'self_attention': attn,
        'num_workers': workers,
        'early_stopping_patience': early_stopping_patience,
        'guidance_scale': guidance_scale,
        'conditional': conditional,
        'num_attributes': num_attributes,
        'cfg_mask_prob': cfg_mask_prob,
        'seed': seed,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'use_importance_sampling': use_is,
        'eps': eps, 
        'likelihood_weighting': likelihood_weighting
        }
    
    

    # 4. Initialize Modules and DataLoaders
    ldm_module, train_loader, val_loader, test_loader = setup(cfg, data_path, device)
    
    # --- W&B and Logging Setup ---
    self_attention = cfg['self_attention']
    lr_str = str(lr).replace('.', '')
    hyper_suffix = f"T{N_timesteps}_LR{lr_str}_E{epochs}"

    if self_attention:
        hyper_suffix += "_SA_CFG"
    if resume_ckpt_cli:
        hyper_suffix += "_RESUME_CFG"



    # The WandbLogger integrates logging with the PL Trainer
    wandb_logger = WandbLogger(project = "LDM Training",
        name=f"{model_type}_{hyper_suffix}", config=cfg)       
        #project=os.getenv("WANDB_PROJECT", "LDM Training"), 
        #log_model="all")
    
    # wandb_logger.experiment.log({"config_forward": cfg["ForwardConfig"]})

    # 6. Checkpoint Callback (Saves model states)
    # AIP_MODEL_DIR is the standard Vertex AI output path (e.g., gs://bucket/output_ldm_pl/)
    checkpoint_path = os.getenv("AIP_MODEL_DIR", './checkpoints/')
    interim_save_dir = os.path.join(checkpoint_path, 'interim')
    os.makedirs(interim_save_dir, exist_ok=True)
    timestep_curr = datetime.now().strftime("%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        dirpath=interim_save_dir, 
        filename=f"ldm-epoch{{epoch:02d}}-val_loss{{val_loss:.4f}}-{timestep_curr}",
        monitor='val_loss',
        mode='min',
        every_n_epochs=50,
        save_top_k=1, 
        save_last=False
        #save_on_train_epoch_end=True
    )

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
        total_epochs = args.epochs if args.epochs else cfg['epochs']
        remaining_epochs = max(total_epochs - last_epoch_done, 0)
        print(f"[INFO] Trainer will run {remaining_epochs} additional epochs.")

    else: 
        remaining_epochs = args.epochs if args.epochs else cfg['ForwardConfig']['epochs']
   
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # 8. Initialize Trainer (Optimized for T4/GCP Cost Saving)
    trainer = Trainer(
        logger=wandb_logger,
        accelerator=device_str, #'cpu' or 'cuda'
        devices="auto",                      # Use 1 T4 GPU
        # accumulate_grad_batches=2,
        max_epochs=epochs,
        # precision="16-mixed",           # CRUCIAL: Enables Mixed Precision for speed and VRAM savings on T4
        callbacks=[checkpoint_callback], #, early_stopping],
        limit_train_batches=0.1, 
        limit_val_batches=0.1
    )

    trainer.fit(ldm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

#     study = optuna.create_study(
#     direction="minimize",
#     sampler=optuna.samplers.TPESampler(),
#     pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
# )

#     study.optimize(make_objective(cfg), n_trials=10)
#     print("BEST TRIAL:")
#     print(study.best_trial.value)
#     print(study.best_trial.params)
        
    # 9. Cleanup
    wandb.finish()

    final_save_dir = os.path.join(checkpoint_path, 'weights')


    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        
    actual_epochs = epochs #trainer.current_epoch 
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
    #python -m train --data-path="./data" --epochs=30