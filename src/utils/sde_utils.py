import torch
from .subVP_SDE import subVP_SDE
from .subVP_forward import ForwardProcess

def calculate_importance_sampling_probabilities(sde_model, N_timesteps, device):
    """
    Calcola il tensore di probabilità p_IS per l'Importance Sampling.
    p(t) ∝ g(t)^2 / λ_orig(t)
    """
    T_max = 1.0
    epsilon = 1e-8 # Per stabilità numerica (evitare divisioni per zero)
    
    # 1. Crea il vettore di timestep continui da [eps, 1.0]
    timesteps = torch.linspace(epsilon, T_max, N_timesteps, device=device)
    
    # 2. Calcola i pesi necessari (g(t)^2 e λ_orig(t))
    g_squared = sde_model.get_g_squared(timesteps)
    alpha_original = sde_model.get_alpha_original(timesteps)
    
    # 3. Calcola il peso non normalizzato p(t) ∝ g(t)^2 / λ_orig(t)
    # add epsilon to avoid 0 division
    sampling_weights = g_squared / (alpha_original + epsilon)
    
    # 4. Converting to probabilities
    probabilities = sampling_weights / torch.sum(sampling_weights)
    
    return probabilities

