from typing import Optional, Tuple

import torch
from subVP_SDE import subVP_SDE
from Configurations import ForwardConfig

class ForwardProcess:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N

    @torch.no_grad()
    def get_noised_latents(z0: torch.Tensor, t: float = None, final: bool = False, eps: float = 1e-5, closed_formula : bool = True, steps: int = 500, seed: int = 42 sde_cfg: ForwardProcess = ForwardProcess()) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return noised latents z_t, along with the exact epsilon used and std(t).
        
        Inputs:
        z0: encoded latents. Device and dtype define outputs.
        t: scalar in (0, 1). If None and final=False, defaults to 0.5 for a mid-horizon corruption level.
        final: if True, overrides t and uses t = 1 - eps to avoid t=1 exactly for numerical stability when computing σ(t).
        eps: small offset so that final-time evaluation uses 1 - eps instead of 1.0. Prevents sqrt(1 - exp(…)) from degenerating.
        sde_cfg: ForwardProcess instance carrying beta schedule and N. Used to build a subVP_SDE with matching parameters.

        Operations:
        1. Builds subVP_SDE(beta_min, beta_max, N) on the same device as z0.
        2. Broadcasts scalar t to a batch vector (B,) for the SDE call.
        3. Calls closed-form perturbation: z_t = μ(t|z0) + σ(t) * ε, where ε ~ N(0, I) if not supplied internally by subVP_SDE.
        4. Returns (z_t, ε, σ(t)), where σ(t) has shape (B,).

        Notes:
        - Deterministic given z0, t, and a fixed epsilon.
        - Useful for reproducible corruption by reusing returned epsilon.
        """
        if final:
            t_val = 1.0 - float(eps)
        else:
            t_val = 0.5 if t is None else float(t)

        # Building the SDE on the same device of the latent vector
        sde = subVP_SDE(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max, N=sde_cfg.N)
        
        t_tensor = torch.full((z0.size(0),), t_val, device=z0.device, dtype=z0.dtype)

        if closed_formula:
            z_t, epsilon, std = sde.perturb_closed(z0, t_tensor)
        else:
            z_t, epsilon, std = sde.perturb_simulate_path(z0, t_tensor, steps = steps, seed = seed)
        
        return z_t, epsilon, std

    @torch.no_grad()
    def main():
        cfg = ForwardConfig()
        z0 = torch.load(cfg.input_path, map_location="cpu")
    
        sde_cfg = ForwardProcess(cfg.beta_min,cfg.beta_max, cfg.N)
    
        z_t, epsilon, std = get_noised_latents(z0, t = cfg.t, final = cfg.final, eps = cfg.eps, closed_formula = cfg.closed_formula, steps = cfg.N, seed = cfg.seed, sde_cfg = sde_cfg)
    
        torch.save(z_t, cfg.output_path)