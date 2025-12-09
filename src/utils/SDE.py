# previous name subVP_SDE

import torch
import numpy as np
from typing import Callable, Tuple
import torch.nn as nn
import math
#--------------------------------------------------


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1
    
    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion
    
    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std
    
    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max
    
    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
    
    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G

# ------------------ ADDED BY ME --------------------------
    def sigma_t(self, t: torch.Tensor):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma

    def diffusion_coeff(self, t: torch.Tensor):
        sigma = self.sigma_t(t)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return diffusion_coeff
        
    def perturb_closed(self, x_0: torch.Tensor, t):
        mean, std = self.marginal_prob(x_0, t)
        noise = torch.randn_like(x_0)
        x_t = mean + std[:,None, None, None] * noise
        return x_t, noise, std

# reverse SDE for Euler-Maruyama
    def reverse_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float, scores: torch.Tensor, gen: torch.Generator = None) -> torch.Tensor:
        """
        Euler - Marayuama method:
        
        dx = [ -1/2 β(t) x - g(t)^2 sθ(x,t) ] dt + g(t)sqrt(|dt|)z
        where z ~ N(0, I)

        Update:
        x <- x + dx
        t <- t + dt
        """
        sigma_t = self.sigma_t(t)
        
        # drift = torch.zeros_like(x)
        diffusion_coeff = self.diffusion_coeff(t)
        
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        
        dx = (- (diffusion_coeff[:, None, None, None] ** 2) * scores) * dt + diffusion_coeff * noise 
        
        x_ret = x + dx
        return x_ret

#----------------------------- BEFORE
# class SDE:
#     def __init__(self, beta_min: float =0.1, beta_max: float =20, N: int =1000, schedule: str ="linear"):
#         """Construct the sub-VP SDE

#         Args:
#         beta_min: value of beta(0)
#         beta_max: value of beta(1)
#         N: number of discretization steps
#         schedule: to apply different type of noise scheduler

#         Attributes:
#         beta_0: minimum noise scale at t=0 for the linear schedule.
#         beta_1: maximum noise scale at t=1 for the linear schedule.
#         N: stored grid size, which is usually not used by closed-form routines below.
#         schedule: noise scheduler identifier
#         """
#         self.beta_0 = beta_min
#         self.beta_1 = beta_max
#         self.N = N
#         if schedule not in ('linear', 'exponential'):
#             raise ValueError("Schedule must be 'linear' or 'exponential'")
#         self.schedule = schedule

#         if self.schedule == "exponential":
#             self._k = float(torch.log(torch.tensor(self.beta_1/self.beta_0))

#     def beta(self, t: torch.Tensor) -> torch.Tensor:
#         if self.schedule == "linear":
#             return self.beta_0 + t * (self.beta_1 - self.beta_0)

#         if self.schedule == "exponential":
#             k = t.new_tensor(self._k)
#             return t.new_tensor(self.beta_0) * torch.exp(k * t)
    
#     def B(self, t: torch.Tensor) -> torch.Tensor:
#         """Compute B(t) = ∫_0^t β(s) ds for the chosen schedule."""
#         if self.schedule == "linear":
#             return t * self.beta_0 + 1/2 * t**2 * (self.beta_1 - self.beta_0)
        
#         if self.schedule == "exponential":
#             k = t.new_tensor(self._k)
#             beta0 = t.new_tensor(self.beta_0)
#             return (beta0/k) * (torch.exp(k * t) - 1)
        
#         raise ValueError("Error in scheduler setting.")
    
#     def get_g_squared(self, t: torch.Tensor) -> torch.Tensor:
#         """
#         Computes the coefficient g(t)**2 which is specifically for subVP SDE
#         g(t) is used in the SDE definition as the diffusion coefficient squared.
#         g(t)^2 = β(t)[1 - exp(-2∫_0^t β(s)ds)]
#         """
#         beta_t = self.beta(t)
#         B_t = self.B(t)
#         discount = 1.0 - torch.exp(-2.0 * B_t)
#         g_squared = beta_t * discount
        
#         return g_squared

#     # Compute the DDPM-style weight
#     def get_alpha_original(self, t: torch.Tensor) -> torch.Tensor:
#         """
#         Computes alpha(t) = 1 - (exp(-∫_0^t β(s) ds))
#         """
#         B_t = self.B(t)
#         alpha_t = 1 - torch.exp(-B_t)
#         return alpha_t # changed from squared
            
#     # Instanteneous SDE coefficients
#     def subVP_sde(self, x, t):
#         """Returns instantaneous coefficients of the SDE evaluated at (x,t).
#         This function do not integrate but it provides the per-time drift and diffusion values.

#         Args:
#         x: (B,C,H,W), t: (B,)
        
#         Details:
#         beta(t) = beta_0 + t * (beta_1 - beta_0)
#         B_t = ∫_0^t beta(s) ds
#         discount for subVP SDE := 1 - exp(-2 * ∫_0^t beta(s) ds)
#         g(t) = sqrt(beta(t) * discount)
#         """
#         beta_t = self.beta(t)
#         drift = -0.5 * beta_t[:, None, None, None] * x
#         diffusion = torch.sqrt(self.get_g_squared(t))
        
#         return drift, diffusion

#     # Closed form marginal
#     def mean_coeff(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         return torch.exp(-0.5 * self.B(t))

#     def var(self, t: torch.Tensor) -> torch.Tensor:
#         s = 1 - torch.exp(-self.B(t))
#         return s**2
    
#     def marginal_prob_subvp(self, x0: torch.Tensor, t: torch.Tensor):
#         """
#         Closed form X_t | X_0 for subVP with the chosen schedule.
#         mean = exp(-0.5 B(t)) * x0
#         std  = 1 - exp(-B(t))         (note: no sqrt for subVP)
#         """
#         mean_coeff = self.mean_coeff(t)
#         std = torch.sqrt(self.var(t))
#         mean = mean_coeff[:, None, None, None] * x0  # (B,C,H,W)
#         return mean, std

#     # Closed form forward perturbation
#     def perturb_closed(self, x_0: torch.Tensor, t, noise = None):
#         """Sample X_t by perturbing X_0 with gaussian noise

#         Operation:
#         1. Compute closed-form mean and std of X_t | X_0.
#         2. Draw epsilon ~ N(0, I) with the same shape as x_0 if not provided.
#         3. Return x_t = mean + std * epsilon, along with epsilon and std.

#          Notes:
#           - Fixed x_0, t, and noise.
#           - Suitable for training score/ε-predictor networks with known std.
          
#         Args:
#         x_0: (B,C,H,W), t:(B,)"""
        
#         mean, std = self.marginal_prob_subvp(x_0, t)
#         if noise is None:
#             noise = torch.randn_like(x_0)
#         x_t = mean + std[:,None, None, None] * noise
#         return x_t, noise, std

#     #Forward Euler - Maruyama simulation
#     def perturb_simulate_path(self, x_0: torch.Tensor, t_end: float = 1.0, steps: int = 500, seed: int = 42, eps: float = 1e-12):
#         """Sample X_t by perturbing X_0 with gaussian noise at time t

#         Operation:
#         1. Compute simulate path for of X_t | X_t-1 and updating X_t values for steps time
#         2. omputing the mean and std at time t
#         3. Calculating the implied eps

#          Notes:
#           - With fixed x_0, t, and noise.
#           - Suitable for training score/ε-predictor networks with known std.
          
#         Args:
#         x_0: (B,C,H,W), t:(B,)"""
#         # t_scalar = float(t_end)
        
#         device = x_0.device
#         dtype = x_0.dtype
#         cnt = x_0.shape[0]
    
#         gen = torch.Generator(device = device).manual_seed(seed)

#         t_grid = torch.linspace(0.0, float(t_end), steps + 1, device = device, dtype = dtype)
#         x = x_0.clone()
        
#         for k in range(steps):
#             t_k = t_grid[k].expand(cnt)
#             dt = (t_grid[k+1] - t_grid[k]).item() # we return a scalar value
#             drift, diffusion = self.subVP_sde(x, t_k)
#             # diffusion = torch.sqrt(diffusion)
#             noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) # we generate Gaussian Noise, with same device and dtype as x
#             x = x + drift * dt + diffusion[:, None, None, None] * (dt ** 0.5) * noise #sqrt(dt) is needed because it works as stabilizing term for the variance
        
#         t_tensor = torch.full((cnt,), float(t_end), device = device, dtype = dtype)
#         mean_t, std_t = self.marginal_prob_subvp(x_0, t_tensor)
#         eps_implied = (x - mean_t) / (std_t[:, None, None, None] + 1e-12) #noise tensor
#         return x, eps_implied, std_t

#     # score target for likelihood-weighted DSM

#     # reverse SDE for Euler-Maruyama
#     def reverse_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float, scores: torch.Tensor, gen: torch.Generator = None) -> torch.Tensor:
#         """
#         Euler - Marayuama method:
        
#         dx = [ -1/2 β(t) x - g(t)^2 sθ(x,t) ] dt + g(t)sqrt(|dt|)z
#         where z ~ N(0, I)

#         Update:
#         x <- x + dx
#         t <- t + dt
#         """
#         beta_t = self.beta(t)
#         g2 = self.get_g_squared(t)
#         drift = (-0.5 * beta_t[:, None, None, None] * x) - (g2[:, None, None, None] * scores)
#         noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
#         x_ret = x + drift * dt + torch.sqrt(g2  * abs(dt))[:, None, None, None] * noise
#         return x_ret

#     def probability_flow_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float, scores: torch.Tensor):
#         """
#         Deterministic PF-ODE with Euler step:
#         dx = [ -1/2 β(t) x  - 1/2g(t)^2 sθ(x,t) ] dt
#         """

#         beta_t = self.beta(t)
#         g2 = self.get_g_squared(t)
#         drift = (-0.5 * beta_t[:, None, None, None] * x) - (0.5 * g2[:, None, None, None] * scores)
#         x_ret = x + drift * dt
#         return x_ret
