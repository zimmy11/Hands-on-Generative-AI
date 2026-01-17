import torch
#from  import VESDE
from .WIP_processes import Diffusion_Processes 
from .WIP_SDE import VESDE, SubVPSDE, VPSDE
#from .subVP_forward import ForwardProcess

def calculate_importance_sampling_probabilities(sde_process, N_timesteps, device):
    """
    Closed-form version:
      p_IS(t) ∝ g(t)^2 / λ_orig(t)
    where (Song-style) λ_orig(t) = std(t)^2 of the marginal.

    - VE:     g^2 = diffusion_coeff(t)^2, λ_orig = sigma_t(t)^2
    - VP:     g^2 = beta(t),              λ_orig = 1 - exp(-B(t))
    - subVP:  g^2 = beta(t)*(1-exp(-2B)), λ_orig = (1-exp(-B))^2
    """
    eps = 1e-5
    t = torch.linspace(eps, 1.0 - eps, N_timesteps, device=device)
    
    # Compute g^2 and lambda_orig in closed form
    if isinstance(sde_process, VESDE):
        g_squared = sde_process.diffusion_coeff(t) ** 2
        lambda_orig = sde_process.sigma_t(t) ** 2  # std^2

    elif isinstance(sde_process, VPSDE):
        # g^2 = beta(t)
        g_squared = sde_process.beta(t)
        # std^2 = var(t) = 1 - exp(-B(t))
        lambda_orig = 1.0 - torch.exp(-sde_process.B(t))

    elif isinstance(sde_process, SubVPSDE):
        beta_t = sde_process.beta(t)
        B_t = sde_process.B(t)
        # g^2 = beta(t) * (1 - exp(-2B(t)))
        g_squared = beta_t * (1.0 - torch.exp(-2.0 * B_t))
        # std^2 = var(t) = (1 - exp(-B(t)))^2
        lambda_orig = (1.0 - torch.exp(-B_t)) ** 2

    else:
        raise TypeError(f"Unsupported SDE type: {type(sde_process)}")

    weights = g_squared / (lambda_orig + eps)
    probs = weights / weights.sum()
    return probs
