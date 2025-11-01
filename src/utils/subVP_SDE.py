import torch
import numpy as np

class subVP_SDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    def sde(self, x, t):
        """Returns instantaneous coefficients of the SDE, therefore simulate stepwise
        Args:
        x: (B,C,H,W), t: (B,) """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x #because x is (B, C, H, W), where B is the batch size
        discount = 1.0 - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2) #development of the integral beta(s) ds
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """Closed form for x_t|x_0 ~ N (mean_coeff, std**2 * I)
        Args:
        x0: (B,C,H,W), t: (B,)
        Returns mean (B,1,1,1)*x0 and std (B,)"""
        
        log_mean_coeff = -0.5 * self.beta_0 * t + -0.25 * (t ** 2) * (self.beta_1 - self.beta_0) #log exp(-1/2 * integral ( beta(s) ds)
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x_0
        std = torch.sqrt(1.0 - torch.exp(2 * log_mean_coeff)) #double check
        return mean, std

    def perturb(self, x_0, t, noise = None):
        """Draw x_t and return (x_t, noise, std)
        Args:
        x_0: (B,C,H,W), t:(B,)"""
        
        mean, std = self.marginal_prob(x_0, t)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = mean + std[:,None, None, None] * noise
        return x_t, noise, std
            