# /modules/ldm_module.py

import torch
import pytorch_lightning as pl
from torch import nn
from typing import Optional
from src.utils.sde_utils import *
import os


#from src.models.unet_model import UNet

# Assume these are imported from sde_utils and unet
# from .sde_utils import ForwardProcess, subVP_SDE
# from .unet import UNet

class LDMLightningModule(pl.LightningModule):
    def __init__(self, unet_model, forward_process, vae_encoder, vae_decoder, hparams, cfg):
        super().__init__()
        
        # Save Hyperparameters to W&B/Logger
        self.save_hyperparameters(hparams) 
        
        # Models and Components
        self.unet = unet_model
        self.forward_process = forward_process # Instance of ForwardProcess
        self.criterion = nn.MSELoss(reduction='none') # Loss must be 'none' for per-sample weighting
        
        # VAE Encoder Function (defined in vae_utils)
        self.encode_latents = vae_encoder
        self.decode_latents = vae_decoder

        
        # Config Params
        self.lr = hparams['learning_rate']
        self.vae_scale_factor = hparams['vae_scale_factor']
        self.n_timesteps = hparams['n_timesteps'] # N for IS calculation
        self.cfg = cfg 


    def forward(self, x_t, t):
        """U-Net prediction of epsilon."""
        return self.unet(x_t, t)

    def _get_weighted_loss(self, batch, is_probabilities: Optional[torch.Tensor] = None):
        """Core logic for sampling, corrupting, predicting, and weighting the loss."""
        
        device = self.device # PL handles device placement
        
        # Debug: Print input batch stats
        # if self.global_step % 100 == 0: # Stampa solo ogni 100 step per non intasare
        #     print(f"\n[DEBUG Step {self.global_step}] Input Batch Stats:")
        #     print(f"  Min: {batch.min().item():.4f}, Max: {batch.max().item():.4f}, Mean: {batch.mean().item():.4f}")
        #     if batch.min() < -1.1 or batch.max() > 1.1:
        #         print("  WARNING: Input data seems out of range [-1, 1]!")


        # 1. Encode Data (x_0) and Apply VAE Scale Factor
        x_start_pixels, labels = batch # Assumes Dataloader yields pixel tensor
        
        #print(f"1. [INPUT] Pixels Shape: {x_start_pixels.shape}")

        with torch.no_grad():

            x_start_latents = self.encode_latents(x_start_pixels) * self.vae_scale_factor

        #print(f"1. [INPUT] Pixels Shape after encoding: {x_start_latents.shape}")


        if self.global_step % 100 == 0:
            var_lat = x_start_latents.var()
            mean_lat = x_start_latents.mean()
            #print(f"  Latents (scaled): Mean={mean_lat:.4f}, Var={var_lat:.4f}")
            if var_lat < 0.5 or var_lat > 2.0:
                print(f"  WARNING: Latent Variance is {var_lat:.4f}. Expected ~1.0. Check VAE Scale Factor!")

        batch_size = x_start_latents.shape[0]

        is_probabilities = None
        # 2. Sample time (t) using Importance Sampling (IS) or Uniform
        if is_probabilities is not None:
            # Importance Sampling (using the pre-calculated tensor)
            print("Tensor Is Probabilities", is_probabilities)
            indices = torch.multinomial(is_probabilities, num_samples=batch_size, replacement=True)
            t = (indices.float() / self.n_timesteps).to(device)
            # probs_sampled = is_probabilities[indices]
            # print(f"   Selected Indices: {indices[:].tolist()}...") 
            # print(f" Indices shape: {indices.shape}")
            # print(f" Probs Sampled: {probs_sampled}")
            # print(f"   Probabilities of selected t: Min={probs_sampled.min():.2e}, Max={probs_sampled.max():.2e}")
            # p_t = is_probabilities[indices].to(device)
            
            # # Importance Sampling Weight: 1 / (N * p(t))
            # # Dobbiamo dividere per la probabilità di aver scelto questo t per rendere la stima unbiased
            # importance_weight = 1.0 / (p_t * self.n_timesteps + 1e-10)
            # importance_weight = importance_weight[:, None, None, None]
        else: 
            # Uniform Sampling (Fallback/Plain Likelihood Weighting)
            t = torch.rand(batch_size, device=device)
            #print(f"t: {t} \n") 
            #importance_weight = torch.ones(batch_size, device=device)

        # Call the corrected method (z0, t, noise)
        x_t, _, epsilon_true  = self.forward_process.forward_process(x_start_latents, t)
        
        #print(f"2. [NOISING] Latents Shape: {x_t.shape} at t shape: {t.shape}")
        #print(f"   Epsilon True Shape: {epsilon_true.shape}, Std Shape: {std.shape}")
        print(f"   t Sampled: Min={t.min().item():.4f}, Max={t.max().item():.4f}")
        # 4. Network prediction (epsilon_pred)
        score_pred_scaled = self(x_t, t)
        score_true_scaled = - epsilon_true
        #print(f"3. [PREDICTION] Epsilon Pred Shape: {epsilon_pred.shape}")

        # --- DEBUG 3: Prediction vs Target ---
        # if self.global_step % 100 == 0:
        #     mse_err = self.criterion(epsilon_pred, epsilon_true)
            # print(f"  Target Epsilon: Mean={epsilon_true.mean().detach().cpu().item():.4f}, Std={epsilon_true.std().detach().cpu().item():.4f}")
            # print(f"  Pred Epsilon:   Mean={epsilon_pred.mean().detach().cpu().item():.4f}, Std={epsilon_pred.std().detach().cpu().item():.4f}")
            # print(f"  Current MSE: {mse_err.mean().detach().cpu().item():.6f}")
            
            # Check for NaNs
            # if torch.isnan(epsilon_pred).any():
            #     print("  CRITICAL: NaN detected in epsilon_pred!")
            #     import pdb; pdb.set_trace() # Ferma tutto se c'è NaN

        # 5. Calculate Per-Sample Loss (MSE: ||epsilon_pred - epsilon_true||^2)
        per_sample_loss = self.criterion(score_pred_scaled, score_true_scaled)
        print(f"4. [LOSS] Per-sample loss shape (before reduction): {per_sample_loss.shape}")
        print(f"   Per-sample loss stats: Min={per_sample_loss.min().item():.4f}, Max={per_sample_loss.max().item():.4f}, Mean={per_sample_loss.mean().item():.4f}")


        # 6. Likelihood Weighting (λ(t) = g(t)^2)
        # g_squared_tensor = sde.get_g_squared(t)
        # print(f"5. [WEIGHTING] g(t)^2 shape: {g_squared_tensor.shape}, Stats: Min={g_squared_tensor.min().item():.4f}, Max={g_squared_tensor.max().item():.4f}, Mean={g_squared_tensor.mean().item():.4f}")
        # # Reshape for broadcasting (B, 1, 1, 1)
        # weighting_factor = g_squared_tensor[:, None, None, None] 
        # print(f"   Weighting factor shape (after unsqueeze): {weighting_factor.shape}")
        # weighting_factor = sde.get_alpha_original(t)**2
        # weighting_factor = weighting_factor[:, None, None, None]
        #var_t = std**2
        #weighting_factor = g_squared_tensor[:, None, None, None]      
        #   
        # Total Weighted Loss (L(t) * g(t)^2)
        weighted_loss = per_sample_loss# * weighting_factor #* importance_weight
        # print(f"   Weighted loss shape (before reduction): {weighted_loss.shape}")
        
        # Final batch loss (torch.mean over the batch)
        final_loss = torch.mean(weighted_loss)
        print(f"   Final loss (after mean): {final_loss.item():.6f}")
        
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
    

    # def on_train_epoch_end(self):
    #         """
    #         Callback executed at the end of every training epoch.
    #         We use this to generate and save sample images to monitor progress.
    #         """
            
    #         # 1. Define how often to save (e.g., every 20 epochs)
    #         # We add 1 because epochs are 0-indexed (0, 1, 2...)
    #         save_interval = 20
            
    #         if (self.current_epoch + 1) % save_interval == 0:
                
    #             # --- A. Setup Sampling Configuration ---
    #             # We generate a small batch (e.g., 4 images) to save time.
    #             n_samples = 4 
    #             # Define shape: (Batch, Channels, Height, Width) -> (4, 4, 32, 32)
    #             # Ensure spatial dim matches your UNet input (32x32 latents = 256x256 pixels)
    #             shape = (n_samples, 4, 32, 32) 
                


    #             print(f"\n[Epoch {self.current_epoch}] Generating preview samples...")
                
    #             # --- B. Generate Latents ---
    #             with torch.no_grad():
    #                 # Run the reverse SDE to get latent vectors z_0
    #                 z_gen = self.forward_process.sample_reverse(self.cfg, self.unet)
                
    #             # --- C. Decode Latents to Pixels ---
    #             # IMPORTANT: We must divide by the scale factor to restore original VAE variance
    #             z_gen = z_gen / self.vae_scale_factor 
                
    #             with torch.no_grad():
    #                 # Decode: Latents (4, 32, 32) -> Pixels (3, 256, 256)
    #                 # Assuming vae_decoder returns a class with .sample (like Diffusers AutoencoderKL)
    #                 img_gen = self.decode_latents(z_gen)s

    #             # --- D. Denormalize Image ---
    #             # Convert from [-1, 1] range back to [0, 1] for saving
    #             img_gen = (img_gen / 2 + 0.5).clamp(0, 1)

                
    #             # Define directory path
    #             save_dir = "image_training"
                
    #             # Create directory if it does not exist
    #             if not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
    #                 print(f"Created directory: {save_dir}")

    #             # Construct filename: {run_name}_epoch_{number}.png
    #             filename = f"{self.run_name}_epoch_{self.current_epoch}.png"
    #             full_path = os.path.join(save_dir, filename)
                
    #             # Save the grid
    #             save_image(img_gen, full_path, nrow=2) # 2x2 grid
                
    #             print(f"Preview saved to: {full_path}") 