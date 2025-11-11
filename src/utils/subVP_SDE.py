import torch
import numpy as np

class subVP_SDE:
    def __init__(self, beta_min: float =0.1, beta_max: float =20, N: int =1000, schedule: str ="linear"):
        """Construct the sub-VP SDE

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        schedule: to apply different type of noise scheduler

        Attributes:
        beta_0: minimum noise scale at t=0 for the linear schedule.
        beta_1: maximum noise scale at t=1 for the linear schedule.
        N: stored grid size, which is usually not used by closed-form routines below.
        schedule: noise scheduler identifier
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        if schedule not in ('linear', 'exponential'):
            raise ValueError("Schedule must be 'linear' or 'exponential'")
        self.schedule = schedule

        if self.schedule == "exponential":
            self._k = float(torch.log(torch.tensor(self.beta_1/self.beta_0)))

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "linear":
            return self.beta_0 + t * (self.beta_1 - self.beta_0)

        if self.schedule == "exponential":
            k = t.new_tensor(self._k)
            return t.new_tensor(self.beta_0) * torch.exp(k * t)
    
    def B(self, t: torch.Tensor) -> torch.Tensor:
        """Compute B(t) = ∫_0^t β(s) ds for the chosen schedule."""
        if self.schedule == "linear":
             return t * self.beta_0 + 1/2 * t**2 * (self.beta_1 - self.beta_0)
        
        if self.schedule == "exponential":
            k = t.new_tensor(self._k)
            beta0 = t.new_tensor(self.beta_0)
            return (beta0/k) * (torch.exp(k * t) - 1)
            
    # Instanteneous SDE coefficients
    def subVP_sde(self, x, t):
        """Returns instantaneous coefficients of the SDE evaluated at (x,t).
        This function do not integrate but it provides the per-time drift and diffusion values.

        Args:
        x: (B,C,H,W), t: (B,)
        
        Details:
        beta(t) = beta_0 + t * (beta_1 - beta_0)
        B_t = ∫_0^t beta(s) ds
        discount for subVP SDE := 1 - exp(-2 * ∫_0^t beta(s) ds) = 1 - exp(-2 * beta_0 * t - (beta_1 - beta_0) * t^2)
        g(t) = sqrt(beta(t) * discount)
        """
        beta_t = self.beta(t)
        B_t = self.B(t)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(-2.0 * B_t)
        diffusion = torch.sqrt(beta_t * discount)
        
        return drift, diffusion

    # Closed form formula for linear scheduler
    def marginal_prob_subvp(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Closed form X_t | X_0 for subVP with the chosen schedule.
        mean = exp(-0.5 B(t)) * x0
        std  = 1 - exp(-B(t))         (note: no sqrt for subVP)
        """
        B_t = self.B(t)
        mean_coeff = torch.exp(-0.5 * B_t)
        std = 1.0 - torch.exp(-B_t)
        mean = mean_coeff[:, None, None, None] * x0  # (B,C,H,W)
        return mean, std

    # Closed form perturbation
    def perturb_closed(self, x_0: torch.Tensor, t, noise = None):
        """Sample X_t by perturbing X_0 with gaussian noise

        Operation:
        1. Compute closed-form mean and std of X_t | X_0.
        2. Draw epsilon ~ N(0, I) with the same shape as x_0 if not provided.
        3. Return x_t = mean + std * epsilon, along with epsilon and std.

         Notes:
          - Fixed x_0, t, and noise.
          - Suitable for training score/ε-predictor networks with known std.
          
        Args:
        x_0: (B,C,H,W), t:(B,)"""
        
        mean, std = self.marginal_prob_subvp(x_0, t)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = mean + std[:,None, None, None] * noise
        return x_t, noise, std

    # Euler - Maruyama simulation
    def perturb_simulate_path(self, x_0: torch.Tensor, t_end: float = 0.5, steps: int = 500, seed: int = 42, eps: float = 1e-12):
        """Sample X_t by perturbing X_0 with gaussian noise at time t

        Operation:
        1. Compute simulate path for of X_t | X_t-1 and updating X_t values for steps time
        2. omputing the mean and std at time t
        3. Calculating the implied eps

         Notes:
          - Deterministic for fixed x_0, t, and noise.
          - Suitable for training score/ε-predictor networks with known std.
          
        Args:
        x_0: (B,C,H,W), t:(B,)"""
        # t_scalar = float(t_end)
        
        device = x_0.device
        dtype = x_0.dtype
        cnt = x_0.shape[0]
    
        gen = torch.Generator(device = device).manual_seed(seed)

        t_grid = torch.linspace(0.0, float(t_end), steps + 1, device = device, dtype = dtype)
        x = x_0.clone()
        
        for k in range(steps):
            t_k = t_grid[k].expand(cnt)
            dt = (t_grid[k+1] - t_grid[k]).item() # we return a scalar value
            drift, diffusion = self.subVP_sde(x, t_k)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) # we generate Gaussian Noise, with same device and dtype as x
            x = x + drift * dt + diffusion[:, None, None, None] * (dt ** 0.5) * noise
        
        t_tensor = torch.full((cnt,), float(t_end), device = device, dtype = dtype)
        mean_t, std_t = self.marginal_prob_subvp(x_0, t_tensor)
        eps_implied = (x - mean_t) / (std_t[:, None, None, None] +1e-12) #noise tensor
        return x, eps_implied, std_t
            
            