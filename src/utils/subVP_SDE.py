import torch
import numpy as np

class subVP_SDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps

        Attributes:
        beta_0: minimum noise scale at t=0 for the linear schedule.
        beta_1: maximum noise scale at t=1 for the linear schedule.
        N: stored grid size, which is usually not used by closed-form routines below.
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    def sde(self, x, t):
        """Returns instantaneous coefficients of the SDE evaluated at (x,t).
        This function do not integrate but it provides the per-time drift and diffusion values.

        Args:
        x: (B,C,H,W), t: (B,)
        
        Details:
        beta(t) = beta_0 + t * (beta_1 - beta_0)
        ∫_0^t beta(s) ds = beta_0 * t + 0.5 * (beta_1 - beta_0) * t^2
        discount := 1 - exp(-2 * ∫_0^t beta(s) ds) = 1 - exp(-2 * beta_0 * t - (beta_1 - beta_0) * t^2)
        g(t) = sqrt(beta(t) * discount)
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x #because x is (B, C, H, W), where B is the batch size
        discount = 1.0 - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2) #development of the integral beta(s) ds
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """Closed form for X_t given X_0 = x_O
        
        Distribution:
        X_t|X_0 ~ N (mean_coeff*X_0, std^2 * I)

        Args:
        x0: (B,C,H,W), t: (B,)
        Returns mean (B,1,1,1)*x0 and std (B,)

        Details:
        ∫_0^t beta(s) ds = beta_0 * t + 0.5 * (beta_1 - beta_0) * t^2
        log_mean_coeff = -0.5 * ∫_0^t beta(s) ds
                        = -0.5 * beta_0 * t - 0.25 * (beta_1 - beta_0) * t^2
        mean = exp(log_mean_coeff) * x_0
        std  = sqrt(1 - exp(2 * log_mean_coeff))
        
        """
        
        log_mean_coeff = -0.5 * self.beta_0 * t + -0.25 * (t ** 2) * (self.beta_1 - self.beta_0) #log exp(-1/2 * integral ( beta(s) ds)
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x_0
        std = torch.sqrt(1.0 - torch.exp(2 * log_mean_coeff)) #double check
        return mean, std

    def perturb(self, x_0, t, noise = None):
        """Sample X_t by perturbing X_0 with gaussian noise at time t

        Operation:
        1. Compute closed-form mean and std of X_t | X_0.
        2. Draw epsilon ~ N(0, I) with the same shape as x_0 if not provided.
        3. Return x_t = mean + std * epsilon, along with epsilon and std.

         Notes:
          - Deterministic for fixed x_0, t, and noise.
          - Suitable for training score/ε-predictor networks with known std.
          
        Args:
        x_0: (B,C,H,W), t:(B,)"""
        
        mean, std = self.marginal_prob(x_0, t)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = mean + std[:,None, None, None] * noise
        return x_t, noise, std
            