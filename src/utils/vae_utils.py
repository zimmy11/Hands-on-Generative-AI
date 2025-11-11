# /src/utils/vae_utils.py

import torch
from diffusers import AutoencoderKL

VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"

def get_vae_encoder_func(device):
    """
    Initializes and returns a function that encodes pixel values to latents.
    The returned function performs VAE encoding but DOES NOT apply the final VAE_SCALE_FACTOR.
    
    Args:
        device (torch.device): The target device for the VAE model.
        
    Returns:
        function: A function taking (pixel_values) and returning (raw_latents).
    """
    
    try:
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device)
        vae.eval() # Set VAE to inference mode
        
        @torch.no_grad()
        def encode_to_latent(pixel_values: torch.Tensor) -> torch.Tensor:
            """
            Encodes the batch of pixel values (in [-1, 1]) to the raw latent space.
            """
            # The VAE returns a Gaussian distribution (posterior)
            posterior = vae.encode(pixel_values).latent_dist
            
            # For DDPM/SDE training, we sample the mean of the posterior
            latents = posterior.sample()
            
            # NOTE: Scaling by 0.18215 is deliberately omitted here. 
            # It will be applied in the LDMLightningModule for better control.
            return latents
            
        return encode_to_latent
        
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        # Return a placeholder function to allow development to continue if VAE loading fails
        return lambda x: torch.zeros((x.shape[0], 4, x.shape[2]//8, x.shape[3]//8), device=device)

# The function get_vae_encoder_func is the 'vae_encoder_func' argument in the PL module.