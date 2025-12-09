import torch
from .subVP_SDE import subVP_SDE
from .subVP_processes import DiffusionProcesses 
#from .subVP_forward import ForwardProcess

def calculate_importance_sampling_probabilities(sde_model, N_timesteps, device):
    """
    Calcola il tensore di probabilità p_IS per l'Importance Sampling.
    p(t) ∝ g(t)^2 / λ_orig(t)
    """
    epsilon = 1e-5 # Per stabilità numerica (evitare divisioni per zero)
    T_max = 1.0 - epsilon 

    # 1. Crea il vettore di timestep continui da [eps, 1.0]
    timesteps = torch.linspace(epsilon, T_max, N_timesteps, device=device)
    
    # 2. Calcola i pesi necessari (g(t)^2 e λ_orig(t))
    g_squared = sde_model.get_g_squared(timesteps)
    alpha_original = sde_model.get_alpha_original(timesteps) ** 2
    
    print(f"G-Squared Max: {torch.max(g_squared)}, Min g^2: {torch.min(g_squared)} Avg. g^2: {torch.mean(g_squared)}, Std. g^2: {torch.std(g_squared)}")
    print(f"Alpha Squared Max: {torch.max(alpha_original)}, Min alpha: {torch.min(alpha_original)} Avg. alpha: {torch.mean(alpha_original)}, Std. alpha: {torch.std(alpha_original)}")

    # 3. Calcola il peso non normalizzato p(t) ∝ g(t)^2 / λ_orig(t)
    # add epsilon to avoid 0 division
    sampling_weights = g_squared / (alpha_original + epsilon)
    
    # 4. Converting to probabilities
    probabilities = sampling_weights / torch.sum(sampling_weights)
    
    return probabilities

