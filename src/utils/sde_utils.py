import torch
from .WIP_processes import Diffusion_Processes 
from .WIP_SDE import VESDE, SubVPSDE, VPSDE
# from .subVP_forward import ForwardProcess


def calculate_importance_sampling_probabilities(sde_model, N_timesteps, device):
    """
    Computes the probability tensor for Importance Sampling (IS) over timesteps.

    Formula: p(t) ∝ g(t)^2 / λ_orig(t)

    Args:
        sde_model: SubVPSDE, VESDE, or VPSDE instance with required methods.
        N_timesteps (int): Number of timesteps to discretize [0, 1].
        device (torch.device): Target device (CPU or GPU).

    Returns:
        torch.Tensor: Normalized probability tensor of shape (N_timesteps,)
    """
    epsilon = 1e-5  # Small number for numerical stability to avoid division by zero
    T_max = 1.0 - epsilon  # Maximum normalized time

    # 1. Create a vector of timesteps in [epsilon, 1.0]
    timesteps = torch.linspace(epsilon, T_max, N_timesteps, device=device)

    # 2. Compute the weighting factors
    # g(t)^2: the squared diffusion coefficient
    g_squared = sde_model.get_g_squared(timesteps)

    # α_orig(t)^2: original alpha squared
    alpha_original = sde_model.get_alpha_original(timesteps) ** 2

    # Log some statistics for debugging
    print(f"G-squared | max: {torch.max(g_squared):.6f}, min: {torch.min(g_squared):.6f}, "
          f"mean: {torch.mean(g_squared):.6f}, std: {torch.std(g_squared):.6f}")
    print(f"Alpha^2   | max: {torch.max(alpha_original):.6f}, min: {torch.min(alpha_original):.6f}, "
          f"mean: {torch.mean(alpha_original):.6f}, std: {torch.std(alpha_original):.6f}")

    # 3. Compute unnormalized IS weights: w(t) = g(t)^2 / (α_orig(t)^2 + epsilon)
    sampling_weights = g_squared / (alpha_original + epsilon)

    # 4. Normalize to get a probability distribution over timesteps
    probabilities = sampling_weights / torch.sum(sampling_weights)

    return probabilities
