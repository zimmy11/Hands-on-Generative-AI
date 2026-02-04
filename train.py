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

# Import core utility functions from the project
from src.utils.sde_utils import * 
# from src.utils.subVP_forward import ForwardProcess   # Optional: subVP SDE forward process
from src.utils.utils import setup


def extract_epoch_from_filename(ckpt_filename: str) -> int:
    """
    Extracts the epoch number from a checkpoint filename.
    Example: 'LDM_final_T1000_LR00001_E24_SA_ts20251125_144930.pth' -> 24
    """
    match = re.search(r'_E(\d+)', ckpt_filename)
    if match:
        return int(match.group(1))
    return 0


def main():
    # 1. Parse command-line arguments (for local or cloud training)
    parser = argparse.ArgumentParser(description="PyTorch Lightning LDM Training")
    parser.add_argument('--data-path', type=str, required=True, help='Dataset directory path (can be GCS path for cloud).')
    parser.add_argument('--config-path', type=str, default='./experiments/base_config.yaml', help='Path to YAML config file.')
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='Checkpoint path to resume training.')

    # Optional overrides for YAML config
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
    parser.add_argument('--sde_type', type=str)

    parser.add_argument('--guidance-scale', type=bool)
    parser.add_argument('--conditional', type=int)
    parser.add_argument('--num_attributes', type=int)
    parser.add_argument('--cfg_mask_prob', type=str)

    args = parser.parse_args()

    # 2. Load YAML configuration
    torch.set_float32_matmul_precision('medium')  # Optimized matrix multiplication for training

    print(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    resume_ckpt_cli = args.resume_checkpoint
    resume_ckpt_local = None
    _downloaded_tmp_ckpt = None

    if resume_ckpt_cli:
        print(f"[INFO] Resume checkpoint provided: {resume_ckpt_cli}")
        if not os.path.exists(resume_ckpt_cli):
            print(f"[ERROR] Checkpoint not found at: {resume_ckpt_cli}")
            sys.exit(1)
        resume_ckpt_local = resume_ckpt_cli

    # Helper function to prioritize CLI args over YAML config
    def get_param(key, cli_value):
        return cli_value if cli_value is not None else yaml_config.get(key)

    # Extract parameters from YAML or CLI
    beta_min = get_param('beta_min', args.beta_min)
    beta_max = get_param('beta_max', args.beta_max)
    N_timesteps = get_param('n_timesteps', args.n_timesteps)
    schedule = get_param('schedule', args.schedule)
    seed = get_param('seed', args.seed)

    t_forward = get_param('t_forward', args.t_forward)
    final = get_param('final', args.final)
    eps = get_param('eps', args.eps)
    closed_formula = get_param('closed_formula', args.closed_formula)
    sde_type = get_param('sde_type', args.sde_type)
    t_0 = get_param('t_0', args.t0)
    t_1 = get_param('t_1', args.t1)
    corrector = get_param('corrector', args.corrector)
    n_corr = get_param('n_corr', args.n_corr)
    target_snr = get_param('target_snr', args.target_snr)
    rev_type = get_param('rev_type', args.rev_type)

    epochs = get_param('epochs', args.epochs)
    lr = get_param('learning_rate', args.learning_rate)
    batch_size = get_param('batch_size', args.batch_size)
    model_type = get_param('model', args.model)

    use_is = get_param('use_importance_sampling', args.use_importance_sampling)
    latent_ch = get_param('latent_channels', args.latent_channels)
    image_size = get_param('image_size', args.image_size)
    vae_scale = get_param('vae_scale_factor', args.vae_scale_factor)
    vae_factor = get_param('vae_factor', args.vae_factor)
    val_split = get_param('validation_split_ratio', args.validation_split)
    feats = get_param('features', args.features)
    attn = get_param('self_attention', args.self_attention)
    workers = get_param('num_workers', args.num_workers)
    early_stopping_patience = get_param('early_stopping_patience', args.early_stopping_patience)

    guidance_scale = get_param('guidance_scale', args.guidance_scale)
    conditional = get_param('conditional', args.conditional)
    num_attributes = get_param('num_attributes', args.num_attributes)
    cfg_mask_prob = get_param('cfg_mask_prob', args.cfg_mask_prob)

    # Compute latent spatial dimensions
    latent_h = image_size // vae_factor
    latent_w = image_size // vae_factor
    current_shape = (batch_size, latent_ch, latent_h, latent_w)

    # 3. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Configuration dictionary
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
        'eps': eps
    }

    # 5. Initialize model and dataloaders
    ldm_module, train_loader, val_loader = setup(cfg, None, device)

    # 6. Setup W&B logging
    lr_str = str(lr).replace('.', '')
    hyper_suffix = f"T{N_timesteps}_LR{lr_str}_E{epochs}"
    if attn:
        hyper_suffix += "_SA_CFG"
    if resume_ckpt_cli:
        hyper_suffix += "_RESUME_CFG"

    wandb_logger = WandbLogger(
        project="LDM Training",
        name=f"{model_type}_{hyper_suffix}",
        config=cfg
    )

    # 7. Checkpoint callback
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
    )

    # 8. Handle checkpoint resuming
    ckpt_path_for_trainer = None
    if resume_ckpt_local:
        last_epoch_done = extract_epoch_from_filename(os.path.basename(resume_ckpt_local))
        print(f"[INFO] Resuming from checkpoint: {resume_ckpt_local}")
        try:
            ckpt_dict = torch.load(resume_ckpt_local, map_location="cpu")
            if isinstance(ckpt_dict, dict) and 'state_dict' in ckpt_dict:
                print("[INFO] Loading state_dict into model")
                ldm_module.load_state_dict(ckpt_dict['state_dict'], strict=False)
                ckpt_path_for_trainer = None
            elif isinstance(ckpt_dict, dict) and ('pytorch-lightning_version' in ckpt_dict or 'optimizer_states' in ckpt_dict):
                print("[INFO] Detected Lightning checkpoint, will pass to trainer")
                ckpt_path_for_trainer = resume_ckpt_local
            else:
                ldm_module.load_state_dict(ckpt_dict, strict=False)
                ckpt_path_for_trainer = None
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            ckpt_path_for_trainer = resume_ckpt_local

        total_epochs = args.epochs if args.epochs else cfg['ForwardConfig']['epochs']
        remaining_epochs = max(total_epochs - last_epoch_done, 0)
        print(f"[INFO] Trainer will run {remaining_epochs} more epochs.")
    else:
        remaining_epochs = args.epochs if args.epochs else cfg['ForwardConfig']['epochs']

    # 9. Initialize PyTorch Lightning trainer
    trainer = Trainer(
        logger=wandb_logger,
        accelerator='cuda', 
        devices="auto",
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        limit_train_batches=0.08,
        limit_val_batches=0.08
    )

    # 10. Start training
    print("Starting LDM Training...")
    trainer.fit(ldm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 11. Finish logging
    wandb.finish()

    # 12. Save final model weights
    final_save_dir = os.path.join(checkpoint_path, 'weights')
    os.makedirs(final_save_dir, exist_ok=True)
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_epochs = epochs
    hyper_suffix = re.sub(r'E\d+', f'E{actual_epochs}', hyper_suffix)
    hyper_suffix += f"_ts{current_timestamp}"
    final_model_filename = f"{model_type}_final_{hyper_suffix}.pth"

    final_model_path = os.path.join(final_save_dir, final_model_filename)
    torch.save({'state_dict': ldm_module.state_dict()}, final_model_path)
    print(f"\n[FINAL SAVE] Model saved at: {final_model_path}")

    # Remove temporary checkpoint if any
    if _downloaded_tmp_ckpt and os.path.exists(_downloaded_tmp_ckpt):
        try:
            os.remove(_downloaded_tmp_ckpt)
            print(f"[INFO] Removed temporary checkpoint {_downloaded_tmp_ckpt}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
