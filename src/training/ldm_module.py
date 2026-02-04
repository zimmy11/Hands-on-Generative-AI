# /modules/ldm_module.py

import torch
import pytorch_lightning as pl
from torch import nn
from typing import Optional
from src.utils.sde_utils import *
import torch.nn.functional as F
from src.models.components import EMAModel


class LDMLightningModule(pl.LightningModule):
    def __init__(self, unet_model, forward_process, vae_encoder, vae_decoder, hparams, cfg):
        super().__init__()
        
        # Save Hyperparameters to W&B/Logger
        self.save_hyperparameters(hparams) 
        
        # Models and Components
        self.unet = unet_model
        self.forward_process = forward_process # Instance of ForwardProcess
        self.criterion = nn.MSELoss(reduction = 'none') # Loss must be 'none' for per-sample weighting
        
        # VAE Encoder Function (defined in vae_utils)
        self.encode_latents = vae_encoder
        self.decode_latents = vae_decoder

        
        # Config Params
        self.lr = hparams['learning_rate']
        self.vae_scale_factor = hparams['vae_scale_factor']
        self.n_timesteps = hparams['n_timesteps'] # N for IS calculation
        self.cfg = cfg 
        self.eps = float(cfg['eps'])
        self.ema_model = hparams['ema']

        self.cfg_mask_prob = cfg.get('cfg_mask_prob', 0.1)



    def forward(self, x_t, t, labels):
        """U-Net prediction of epsilon."""
        return self.unet(x_t, t, labels)

    def _get_weighted_loss(self, batch, is_probabilities: Optional[torch.Tensor] = None):
        """Core logic for sampling, corrupting, predicting, and weighting the loss."""
        
        device = self.device # PL handles device placement



        # 1. Encode Data (x_0) and Apply VAE Scale Factor & # Masking for conditional generation
        if self.cfg['conditional'] == True:
            x_start_latents, labels = batch # Assumes Dataloader yields pixel tensor
            batch_size = x_start_latents.shape[0]
            mask = torch.bernoulli(torch.full((batch_size, 1), 1- self.cfg_mask_prob, device=self.device))
            cond_labels = labels * mask
        else:
            x_start_latents, _ = batch
            batch_size = x_start_latents.shape[0]
            cond_labels = None
        

        with torch.no_grad():

            x_start_latents = self.encode_latents(x_start_latents) * self.vae_scale_factor



        if self.global_step % 100 == 0:
            var_lat = x_start_latents.var()
            mean_lat = x_start_latents.mean()
     
        is_probabilities = None
        # 2. Sample time (t) using Importance Sampling (IS) or Uniform
        if is_probabilities is not None:
            # Importance Sampling (using the pre-calculated tensor)
            #print("Tensor Is Probabilities", is_probabilities)
            indices = torch.multinomial(is_probabilities, num_samples=batch_size, replacement=True)
            t = (indices.float() / self.n_timesteps).to(device)
            t = t.clamp(min=self.eps)
            
        else: 
            # Uniform Sampling (Fallback/Plain Likelihood Weighting)
           
            t = torch.rand(batch_size, device=device) * (1. - self.eps) + self.eps


        # Call the corrected method (z0, t, noise)
        x_t, _, epsilon_true  = self.forward_process.forward_process(x_start_latents, t)


        epsilon_pred = self(x_t, t, cond_labels)
        

        # 5. Calculate Per-Sample Loss (MSE: ||epsilon_pred - epsilon_true||^2)
        per_sample_loss = self.criterion(epsilon_pred, epsilon_true)

        weighted_loss = per_sample_loss# * weighting_factor #* importance_weight
        
        # Final batch loss (torch.mean over the batch)
        final_loss = torch.mean(weighted_loss)
        
        return final_loss, final_loss.detach() # Return loss and detached value for logging

    # --- PL Required Methods ---

    def training_step(self, batch, batch_idx):
        
        loss, loss_detached = self._get_weighted_loss(batch, is_probabilities = self.hparams.is_probabilities)
        self.log('train_loss', loss_detached, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # We assume the same IS logic for consistency, but often Validation uses uniform sampling.
        loss, loss_detached = self._get_weighted_loss(batch, is_probabilities=self.hparams.is_probabilities) 
        self.log('val_loss', loss_detached, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    