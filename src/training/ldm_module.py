# /modules/ldm_module.py

import torch
import pytorch_lightning as pl
from torch import nn
from typing import Optional
from src.utils.sde_utils import *


#from src.models.unet_model import UNet

# Assume these are imported from sde_utils and unet
# from .sde_utils import ForwardProcess, subVP_SDE
# from .unet import UNet

class LDMLightningModule(pl.LightningModule):
    def __init__(self, unet_model, forward_process, vae_encoder, hparams):
        super().__init__()
        
        # Save Hyperparameters to W&B/Logger
        self.save_hyperparameters(hparams) 
        
        # Models and Components
        self.unet = unet_model
        self.forward_process = forward_process # Instance of ForwardProcess
        self.criterion = nn.MSELoss(reduction='none') # Loss must be 'none' for per-sample weighting
        
        # VAE Encoder Function (defined in vae_utils)
        self.encode_latents = vae_encoder
        
        # Config Params
        self.lr = hparams['learning_rate']
        self.vae_scale_factor = hparams['vae_scale_factor']
        self.n_timesteps = hparams['n_timesteps'] # N for IS calculation

    def forward(self, x_t, t):
        """U-Net prediction of epsilon."""
        return self.unet(x_t, t)

    def _get_weighted_loss(self, batch, is_probabilities: Optional[torch.Tensor] = None):
        """Core logic for sampling, corrupting, predicting, and weighting the loss."""
        
        device = self.device # PL handles device placement
        
        # 1. Encode Data (x_0) and Apply VAE Scale Factor
        x_start_pixels = batch # Assumes Dataloader yields pixel tensor
        with torch.no_grad():
            x_start_latents = self.encode_latents(x_start_pixels) * self.vae_scale_factor

        batch_size = x_start_latents.shape[0]

        # 2. Sample time (t) using Importance Sampling (IS) or Uniform
        if is_probabilities is not None:
            # Importance Sampling (using the pre-calculated tensor)
            indices = torch.multinomial(is_probabilities, num_samples=batch_size, replacement=True)
            t = (indices.float() / self.n_timesteps).to(device)
        else: 
            # Uniform Sampling (Fallback/Plain Likelihood Weighting)
            t = torch.rand(batch_size, device=device)

        # Call the corrected method (z0, t, noise)
        x_t, epsilon_true, std, sde  = self.forward_process.run_forward(x_start_latents, without_likelihood = True)

        # 4. Network prediction (epsilon_pred)
        epsilon_pred = self(x_t, t)

        # 5. Calculate Per-Sample Loss (MSE: ||epsilon_pred - epsilon_true||^2)
        per_sample_loss = self.criterion(epsilon_pred, epsilon_true)
        
        # 6. Likelihood Weighting (λ(t) = g(t)^2)
        g_squared_tensor = sde.get_g_squared(t)
        
        # Reshape for broadcasting (B, 1, 1, 1)
        weighting_factor = g_squared_tensor[:, None, None, None] 
        
        # Total Weighted Loss (L(t) * g(t)^2)
        weighted_loss = per_sample_loss * weighting_factor
        
        # Final batch loss (torch.mean over the batch)
        final_loss = torch.mean(weighted_loss)
        
        return final_loss, final_loss.detach() # Return loss and detached value for logging

    # --- PL Required Methods ---

    def training_step(self, batch, batch_idx):
        loss, loss_detached = self._get_weighted_loss(batch, self.hparams.is_probabilities)
        self.log('train_loss', loss_detached, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # We assume the same IS logic for consistency, but often Validation uses uniform sampling.
        loss, loss_detached = self._get_weighted_loss(batch, self.hparams.is_probabilities) 
        self.log('val_loss', loss_detached, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)