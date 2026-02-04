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
        self.use_ema = cfg.get('use_ema', True)
        self.dataset_type = hparams.get('dataset_type', 'Celeb')
        if self.use_ema:
            self.ema_model = EMAModel(self.unet, decay=hparams.get('ema_decay', 0.9999))

        self.likelihood_weighting = cfg.get('likelihood_weighting', True)
        #self.ema_model = hparams['ema']

        self.cfg_mask_prob = cfg.get('cfg_mask_prob', 0.1)


    def forward(self, x_t, t, labels, cond_mask = None):
        """U-Net prediction of epsilon."""
        return self.unet(x_t, t, labels, cond_mask)

    def _get_weighted_loss(self, batch, is_probabilities: Optional[torch.Tensor] = None):
        """Core logic for sampling, corrupting, predicting, and weighting the loss."""
        
        device = self.device # PL handles device placement
        

        # 1. Encode Data (x_0) and Apply VAE Scale Factor
        x_start_latents, labels = batch # Assumes Dataloader yields pixel tensor
        batch_size = x_start_latents.shape[0]

        if self.dataset_type == "MNIST":
            # Apply class conditioning mask (cfg_mask_prob) 
            labels = F.one_hot(labels, num_classes=10).float()

        num_attributes = labels.shape[1]



        if self.cfg['conditional'] == True:
            p_uncond = self.cfg_mask_prob
            cond_mask = (torch.rand(batch_size, device=device) >= p_uncond).float()

        else:
            cond_mask = torch.zeros(batch_size, device=device)
        
        if self.dataset_type=="Celeb":
            with torch.no_grad():
                x_start_latents = self.encode_latents(x_start_latents) * self.vae_scale_factor

        
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
            #importance_weight = torch.ones(batch_size, device=device)

        # Call the corrected method (z0, t, noise)
        x_t, _, epsilon_true  = self.forward_process.forward_process(x_start_latents, t)

        epsilon_pred = self(x_t, t, labels, cond_mask=cond_mask)

        
        # 5. Calculate Per-Sample Loss (MSE: ||epsilon_pred - epsilon_true||^2)
        per_sample_loss = self.criterion(epsilon_pred, epsilon_true)
        

        # 6. Likelihood Weighting (λ(t) = g(t)^2)
        if self.likelihood_weighting and self.forward_process.sde_type != 've':
            # For subVP SDE, λ(t) = g(t)^2
            g_squared_tensor = self.forward_process.sde.g_squared(t)
            # # Reshape for broadcasting (B, 1, 1, 1)
            weighting_factor = g_squared_tensor[:, None, None, None] 
        else:
            weighting_factor = torch.ones((batch_size, 1, 1, 1), device=device)
        #   
        # Total Weighted Loss (L(t) * g(t)^2)
        weighted_loss = per_sample_loss * weighting_factor #* importance_weight
        
        # Final batch loss (torch.mean over the batch)
        final_loss = torch.mean(weighted_loss)
        
        return final_loss, final_loss.detach() # Return loss and detached value for logging

    # --- PL Required Methods ---

    def training_step(self, batch, batch_idx):
        
        loss, loss_detached = self._get_weighted_loss(batch, is_probabilities = self.hparams.is_probabilities)
        self.log('train_loss', loss_detached, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.ema_model.update(self.unet)

        return loss

    def validation_step(self, batch, batch_idx):
        # We assume the same IS logic for consistency, but often Validation uses uniform sampling.
        loss, loss_detached = self._get_weighted_loss(batch, is_probabilities=self.hparams.is_probabilities) 
        self.log('val_loss', loss_detached, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    



            
 