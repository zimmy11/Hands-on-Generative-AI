import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import wandb
import os
from tqdm import tqdm
import torchvision

from src.utils.vae_utils import get_vae_encoder_func
from src.utils.sde_utils import Diffusion_Processes, SubVPSDE, VESDE, VPSDE
from src.utils.utils import setup, log_denoising_step_wandb
from src.models.UNet import UNet
from src.models.components import EMAModel


def load_model_from_checkpoint(cfg, checkpoint_path, device):
    """
    Load UNet model and checkpoint state for evaluation.
    Supports Lightning checkpoints with 'state_dict' or EMA models.
    """
    # Initialize UNet architecture
    unet = UNet(
        in_channels=cfg['latent_channels'], 
        model_channels=64,  # default or from cfg
        dropout=0.0, 
        num_attributes=cfg.get('num_attributes', 40)
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state_dict
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}

    # Remove Lightning prefixes like 'unet_model.' or 'ema_model.'
    for k, v in state_dict.items():
        if 'unet_model.' in k:
            new_state_dict[k.replace('unet_model.', '')] = v
        # Uncomment to use EMA weights
        # elif 'ema_model.' in k:
        #    new_state_dict[k.replace('ema_model.', '')] = v
        else:
            new_state_dict[k] = v

    # Load weights (strict=False to ignore extra keys)
    unet.load_state_dict(new_state_dict, strict=False)
    unet.eval()
    return unet


def calculate_nll(model, loader, forward_process, device, vae_encoder, vae_scale_factor):
    """
    Computes approximate NLL (or average loss) on the test set.
    Uses training loss as a proxy for exact SDE likelihood.
    """
    print(">>> Calculating NLL (Test Loss)...")
    total_loss = 0
    num_batches = 0
    loss_fn = nn.MSELoss()  # or the specific training loss

    with torch.no_grad():
        for batch in tqdm(loader, desc="NLL Computation"):
            if isinstance(batch, list):
                x = batch[0].to(device)
                y = batch[1].to(device)
            else:
                x = batch.to(device)
                y = None

            # Encode to latent space
            latents = vae_encoder(x) * vae_scale_factor

            # Sample random time steps for forward process
            t = torch.rand(latents.shape[0], device=device) * forward_process.sde.T
            z_t, t, eps = forward_process.forward_process(latents, t)

            # Predict noise
            if forward_process.conditional and y is not None:
                pred = model(z_t, t, y.float())
            else:
                null_y = torch.zeros((x.shape[0], 40), device=device)
                pred = model(z_t, t, null_y)

            loss = loss_fn(pred, eps)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=2000, help="Number of images to generate for FID")
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['batch_size'] = args.batch_size
    cfg['N'] = cfg.get('n_timesteps', 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize WandB for logging
    wandb.init(
        project="LDM_Testing",
        config=cfg,
        name=f"FID_IS_Evaluation_{os.path.basename(args.checkpoint_path)}"
    )

    # ---------------------------
    # 1. Setup data and model
    # ---------------------------
    _, _, _, test_loader = setup(cfg, args.data_path, device)

    if not os.path.dirname(args.checkpoint_path):
        args.checkpoint_path = os.path.join("checkpoints", "weights", args.checkpoint_path)

    unet = load_model_from_checkpoint(cfg, args.checkpoint_path, device)
    diff_proc = Diffusion_Processes(cfg)
    vae_encoder, _ = get_vae_encoder_func(device)
    vae_scale_factor = cfg.get('vae_scale_factor', 0.18215)

    # ---------------------------
    # 2. Compute NLL (Test Loss)
    # ---------------------------
    nll_score = calculate_nll(
        unet, test_loader, diff_proc, device,
        vae_encoder=vae_encoder, vae_scale_factor=vae_scale_factor
    )
    print(f"Test NLL (Loss): {nll_score:.4f}")
    wandb.log({"test/nll_loss": nll_score})

    # ---------------------------
    # 3. Setup FID and Inception Score metrics
    # ---------------------------
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)

    print(">>> Accumulating real image statistics for FID...")
    real_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Real Images"):
            img = batch[0] if isinstance(batch, list) else batch
            img = img.to(device)
            img_uint8 = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            fid_metric.update(img_uint8, real=True)
            real_count += img.shape[0]
            if real_count >= args.num_samples:
                break

    # ---------------------------
    # 4. Generate images and log denoising steps
    # ---------------------------
    print(f">>> Generating {args.num_samples} fake images for FID/IS...")

    num_batches_gen = (args.num_samples + args.batch_size - 1) // args.batch_size
    shape = (args.batch_size, cfg['latent_channels'], cfg['image_size'], cfg['image_size'])

    for i in tqdm(range(num_batches_gen), desc="Generation"):
        cb_fn = log_denoising_step_wandb if i == 0 else None

        labels = None
        if cfg.get('conditional', False):
            labels = torch.randint(0, 2, (args.batch_size, cfg['num_attributes'])).float().to(device)

        samples = diff_proc.reverse_process(
            model=unet,
            shape=shape,
            device=device,
            labels=labels,
            callback_fn=cb_fn
        )

        samples_uint8 = ((samples + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fid_metric.update(samples_uint8, real=False)
        is_metric.update(samples_uint8)

        if i == 0:
            grid = torchvision.utils.make_grid(samples_uint8[:16], nrow=4)
            wandb.log({"generated_samples/final_batch": wandb.Image(grid)})

    # ---------------------------
    # 5. Compute final FID and IS
    # ---------------------------
    print(">>> Computing final FID and Inception Score...")
    fid_score = fid_metric.compute().item()
    is_score, is_std = is_metric.compute()

    print(f"FID: {fid_score:.4f}")
    print(f"IS: {is_score.item():.4f} +/- {is_std.item():.4f}")

    wandb.log({
        "test/fid": fid_score,
        "test/inception_score": is_score.item(),
        "test/inception_std": is_std.item()
    })

    wandb.finish()


if __name__ == "__main__":
    main()

# Usage:
# python test.py --config-path experiments/base_config.yaml --checkpoint-path checkpoints/weights/ --data-path /path/to/celeba
