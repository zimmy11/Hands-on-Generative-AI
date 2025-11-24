import torch
import torch.nn as nn
from src.utils.subVP_SDE import subVP_SDE
from src.utils.subVP_processes import DiffusionProcesses

class DenoisingScoreMatchingLoss(nn.Module):
    def __init__(self, sde: subVP_sde, N_timesteps: int = 500, epsilon: float = 1e-5):
        """
        Denoising Score Matching Loss with Likelihood Weighting.
        
        Args:
            sde: An instance of the subVP_SDE class.
            epsilon: A small value to avoid numerical instability at t=0.
        """
        super().__init__()
        self.sde = sde
        self.eps = epsilon
        self.N_timesteps = N_timesteps
        self.probabilities = None
        self.time_grid = None
        self.diffusion_processes = DiffusionProcesses(beta_min = self.sde.beta_0, beta_max = self.sde.beta_1, N = self.sde.N)

    def calculate_importance_sampling_probabilities(self, device):
        """
        Calcola il tensore di probabilità p_IS per l'Importance Sampling.
        p(t) ∝ g(t)^2 / λ_orig(t)
        """
        T_max = 1.0
        epsilon = 1e-8 # for numerical stability
        
        # timestep vector
        self.time_grid = torch.linspace(epsilon, T_max, self.N_timesteps, device=device)
        
        # Num and den computation
        g_squared = sde_model.get_g_squared(timesteps)
        alpha_original = sde_model.get_alpha_original(timesteps) ** 2
        
        # Weights computations
        sampling_weights = g_squared / (alpha_original + epsilon)
        
        # Convertion to probabilities
        self.probabilities = sampling_weights / torch.sum(sampling_weights)


    def forward_passage(self, model: nn.Module, x0: torch.Tensor, importance_sampling: bool = True) -> torch.Tensor:
        """
        Computes the weighted score matching loss.
        
        Args:
            model: The U-Net model s_theta(x, t).
            x0: A batch of clean data (B, C, H, W).
            
        Returns:
            Scalar loss value.
        """
        # 1. Sample time t uniformly from [eps, 1]
        # We sample continuous time, as required for SDEs.
        batch_size = x0.shape[0]
        device = x0.device

        if importance_sampling:
            if self.probabilities is None and self.time_grid is None:
                self.calculate_importance_sampling_probabilities(device)

            # Sampling the times
            t_indices = torch.multinomial(self.probabilities, num_samples=batch_size, replacement=True)
            t = self.time_grid[t_indices]
        else:
            t = self.randn(barch_size, device = device)
        
        # Utilizes the closed-form transition kernel defined in subVP_SDE.py
        x_t, z, std = self.sde.perturb_closed(x0, t)

        # Computing the score
        score_prediction = model(x_t, t)

        # Calculating the  Ground Truth Score
        # For Gaussian kernel: score = -z / std
        std_broadcast = std[:, None, None, None]
        target_score = -z / (std_broadcast + 1e-12) # Added tiny constant to avoid 0 division

        # Calculate Likelihood Weighting with lambda(t) = g(t)^2
        g2 = self.sde.get_g_squared(t)
        weights = g2[:, None, None, None]

        # Compute Weighted Squared Error
        # Loss = E[ lambda(t) * || s_theta - target ||^2 ]
        
        losses = torch.square(score_prediction - target_score)
        losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
        weighted_losses = weights.squeeze() * losses
        
        return torch.mean(weighted_losses)