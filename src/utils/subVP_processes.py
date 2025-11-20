from typing import Optional, Tuple
import torch
from src.utils.subVP_SDE import subVP_SDE
from src.utils.Configurations import ForwardConfig, ReverseConfig, LikelihoodConfig
import torch.nn as nn

import time


class DiffusionProcesses:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000, schedule: str = "linear"):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.schedule = schedule

    @torch.no_grad()
    def get_noised_latents(self, z0: torch.Tensor, cfg):
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

        if cfg.final:
            t_val = 1.0 - float(cfg.eps)
        else:
            t_val = 0.5 if cfg.t is None else float(cfg.t)

        # Building the SDE on the same device of the latent vector
        sde = subVP_SDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, N=cfg.N)

        if cfg.closed_formula:
            t_tensor = torch.full((z0.size(0),), t_val, device=z0.device, dtype=z0.dtype)
            z_t, epsilon, std = sde.perturb_closed(z0, t_tensor)
        else:
            # t_tensor = torch.tensor([t_val], device=z0.device, dtype=z0.dtype)
            z_t, epsilon, std = sde.perturb_simulate_path(z0, t_val, steps = cfg.steps, seed = cfg.seed)
        
        return z_t, epsilon, std, sde

    @torch.no_grad()
    def run_forward(self, z0, without_likelihood = True):
        """
        Execute the forward process for the latent noised variables with the parameters passed in Configurations
        """


        if without_likelihood:
            cfg = ForwardConfig()
            z_t, epsilon, std, sde = self.get_noised_latents(
                z0, cfg
            )
            
             #torch.save(z_t, cfg.output_path)

        # else: 
        #     cfg = LikelihoodConfig()

        #     likelihood = self.loglikelihood_subvp_ode(z0, model, cfg)



        return z_t, epsilon, std, sde
    
    def sample_reverse(self, cfg: ReverseConfig, model: nn.Module):
        device = next(model.parameters()).device
        dtype = torch.float32

        sde = subVP_SDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, N=cfg.N)
        
        gen = torch.Generator(device = cfg.device).manual_seed(cfg.seed)

        x = torch.randn(*cfg.shape, device = cfg.device, dtype = cfg.dtype, generator = gen)

        #Time discretization for reversion execution
        t_grid = torch.linspace(cfg.t0, cfg.t1, cfg.steps + 1, device = cfg.device, dtype = cfg.dtype)

        model = model.to(device = device, dtype = dtype).eval()
        
        start_time_fixed = time.time()
        start_time = time.time()
        n_steps = cfg.N//10
        #Reverse process loop
        with torch.no_grad():
            for k in range(cfg.steps):
                if k % n_steps == 0:
                    time_elapsed, start_time = time.time() - start_time, time.time()
                    print(f"Summary stats:\nSteps done: {k}\nTime of last {n_steps} steps: {time_elapsed}\nAverage time of last {n_steps} steps: {time_elapsed/n_steps}\nOverall time:{time.time()-start_time_fixed}")
                t_k = t_grid[k].expand(cfg.shape[0])
                t_k1 = t_grid[k+1].expand(cfg.shape[0])
                dt = (t_k1[0] - t_k[0])
    
                _, std_t = sde.marginal_prob_subvp(x, t_k)

                eps_pred = model(x, t_k)
                scores = - eps_pred / (std_t[:, None, None, None] + 1e-12)
                
                #Predictor
                if cfg.rev_type == "sde":
                    x = sde.reverse_euler_step(x, t_k, dt, scores, gen = gen)
                elif cfg.rev_type == "ode":
                    x = sde.probability_flow_euler_step(x, t_k, dt, scores, gen = gen)
                #Corrector
                # x = sde.corrector_langevin(x, t_k1, scores, n_steps = cfg.n_corr, target_snr = cfg.target_snr, gen = gen)
            
        return x

        
    def run_reverse(self, model:nn.Module):
        cfg = ReverseConfig()
        return self.sample_reverse(cfg, model)


    def loglikelihood_subvp_ode(x0: torch.Tensor, model: nn.Module, lcfg: LikelihoodConfig):
        """
        Integrating the ODE for x_t and simultaneously, the log-density via the instantaneous change of variable formula: ODE x^ = v(x,t)
        d logpt(x_t)/dt = −∇⋅v(x_t,t)
        Therefore:
        log p_0(x_0) = log p_T(x_T) + ∫_0^T ∇⋅v(x_t,t)dt
        """
        device = lcfg.device
        x0 = x0.detach().to(device).clone()
        B_size = x0.size(0)

        #Accumulate the intgral of divergence(v)
        log_det = torch.zeros(B_size, device=device)
        t_grid = torch.linspace(0.0, 1.0, lcfg.steps + 1, device = device)
        
        ode = subVP_SDE(beta_min=lcfg.beta_min, beta_max=lcfg.beta_max, N=lcfg.N)
        
        for k in range(lcfg.steps):
            t = t_grid[k].expand(B_size)
            dt = (t_grid[k+1] - t_grid[k]).item()

            v, div_v = ode.likelihood_euler_step(x, t, model, estimator = lcfg.estimator)

            x = x + v * dt
            log_det = log_det + div_v * dt
        
        log_pT = ode.standard_normal_logprob(x)
        
        return log_pT + log_det

            
        