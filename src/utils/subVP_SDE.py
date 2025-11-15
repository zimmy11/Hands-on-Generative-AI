    import torch
import numpy as np
import typing import Callable, Tuple

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
        
        raise ValueError("Error in scheduler setting.")
    
    def get_g_squared(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the coefficient g(t)**2 which is specifically for subVP SDE
        λ(t) is used in the SDE definition as the diffusion coefficient squared.
        λ(t) = g(t)^2
        """
        beta_t = self.beta(t)
        B_t = self.B(t)
        discount = 1.0 - torch.exp(-2.0 * B_t)
        g_squared = beta_t * discount
        
        return g_squared

    #Compute the DDPM-style weight
    def get_alpha_original(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes alpha(t) = (exp(-∫_0^t β(s) ds))
        """
        B_t = self.B(t)
        alpha_t = torch.exp(-0.5 * B_t)
        return alpha_t ** 2
            
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
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = self.get_g_squared(t)
        
        return drift, diffusion

    # Closed form marginal
    def mean_coeff(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.exp(-0.5 * self.B(t))

    def var(self, t: torch.Tensor) -> torch.Tensor:
        s = 1 - torch.exp(-self.B(t))
        return s**2
    
    def marginal_prob_subvp(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Closed form X_t | X_0 for subVP with the chosen schedule.
        mean = exp(-0.5 B(t)) * x0
        std  = 1 - exp(-B(t))         (note: no sqrt for subVP)
        """
        mean_coeff = self.mean_coeff(t)
        std = torch.sqrt(self.var(t))
        mean = mean_coeff[:, None, None, None] * x0  # (B,C,H,W)
        return mean, std

    # Closed form forward perturbation
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

    #Forward Euler - Maruyama simulation
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
            diffusion = torch.sqrt(diffusion)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) # we generate Gaussian Noise, with same device and dtype as x
            x = x + drift * dt + diffusion[:, None, None, None] * (dt ** 0.5) * noise
        
        t_tensor = torch.full((cnt,), float(t_end), device = device, dtype = dtype)
        mean_t, std_t = self.marginal_prob_subvp(x_0, t_tensor)
        eps_implied = (x - mean_t) / (std_t[:, None, None, None] +1e-12) #noise tensor
        return x, eps_implied, std_t

    # score target for likelihood-weighted DSM

    # reverse SDE for Euler-Maruyama
    def reverse_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float, scores: torch.Tensor, gen: torch.Generator = None) -> torch.Tensor:
        """
        dx = [ -1/2 β(t) x  - g(t)^2 sθ(x,t) ] dt + g(t) dW̄
        """
        beta_t = self.beta(t)
        g2 = self.get_g_squared(t)
        drift = (-0.5 * beta_t[:, None, None, None] * x) - (g2[:, None, None, None] * scores)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        x = x + drift * dt + torch.sqrt(g2  * abs(dt))[:, None, None, None] * noise
        return x

    def probability_flow_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float, scores: torch.Tensor):
        """
        Deterministic PF-ODE with Euler step:
        dx = [ -1/2 β(t) x  - 1/2g(t)^2 sθ(x,t) ] dt
        """

        beta_t = self.beta(t)
        g2 = self.get_g_squared(t)
        drift = (-0.5 * beta_t[:, None, None, None] * x) - (0.5 * g2[:, None, None, None] * scores)
        x = x + drift * dt
        return x
        
    @torch.no_grad()
    def corrector_langevin(self, x: torch.Tensor, t: torch.Tensor, scores: torch.Tensor, n_steps: int = 1, target_snr: float = 0.16, gen: torch.Generator = None):
        """
        x ← x + α s(x,t) + sqrt(2α) z, with α set to reach target SNR per batch.
        """
        for _ in range(n_steps):
            noise = torch.randn(x.shape, device = x.device, dtype = x.dtype, generator = gen)
            # per-sample adaptive step size
            grad_norm = scores.flatten(1).norm(dim=1).clamp_min(1e-12)
            noise_norm = noise.flatten(1).norm(dim=1).clamp_min(1e-12)
            step_size = (target_snr * noise_norm / grad_norm) ** 2 *2.0
            x = x + step_size[:, None, None, None] * scores + torch.sqrt(2.0 * step_size)[:, None, None, None] * noise
        
        return x
        
    #-------------Likelihood computation-----------------------
    def v_field(self, x, t, model):
        """
        Compute a vector field which represents the probability flow ODE:
        dx/dt = v(x,t) = −1/2 β(t)x − 1/2 g(t)^2 s_θ(x,t)
        
        We assumed that:
        - ∇_x log p_t(x) ≈ s_θ(x,t)
        - g(x)^2 is the same as before
        - β(t) is the same as before
        """
        beta_t = self.beta(t).view(-1, *([1] * (x.ndim - 1)))
        # we are:
        #1. taking number of non-batch dimensions: x.ndim-1, ex. (B,C,H,W) is 3
        #2. building a list with many ones : [1] * (x.ndim - 1)
        #3. changing shape like (B,1,1,1) from (-1, previous_list)
        #4. we reshape the tensor with .view()
        g2 = get_g_squared(t).view(-1, *([1] * (x.ndim -1)))
        scores = model(x, t)
        return -0.5 * beta_t * x - 0.5 *g2 * scores

    def standard_normal_logprob(self, x):
        """
        Computing the log-density of a d-dimensional standard normal N(0,1) evaluated at x
        Args:
        - x \in R^d
        - log N(0,I)(x) = -1/2 (∥x∥^2 + d log2π)
        """
        d = x[0].numel() # retrievs the flattened dimensionality per sample
        return -0.5 * (x.view(x.size(0), -1).pow(2).sum(dim=1) + d * math.log(2 * math.pi))
    
    def hutchinson_div_score(self, x, t, model):
        """
        Estimate the diverge of the score: ∇⋅s_θ(x,t) = tr J(x), where J(x) = ∂s_𝜃/∂x.

        
        For Hutchunson's Identity: e ~ N(0,I): tr J = E_e[e^T J e], the trace of the Jacobian is equal to the expected value, over epsilon, of epsilon-transposed J epsilon.
        Args:
        - e ~ N(0,I)
        - ϕ(x)=⟨s_θ(x,t),e⟩ = ∑_i s_i(x,t)e_i
        - jte = ∇_x φ(x), which is J(x)^Te by the chain rule
        - then we can write ⟨(J^Te), e⟩ = e^TJe, which givs us an unbiased estimate of the trace of J
        """
        x = x. requires_grad_(True)
        e = torch.randn_like(x)
        scores = model(x, t)
        jte = torch.autograd.grad((scores * e).sum(), x, create_graph = False, retain_graph = False)[0] # extracting the gradinent since jte has shape (x,)
        div_est = (jte * e).view(x.size(0), -1).sum(dim=1)
        x.requires_grad_(False)
        return div_est

    def likelihood_euler_step(self, x, t, d, model):
        v = self.v_field(x, t, model)
        beta_t = self.beta(t)
        g2 = self.get_g_squared(t)
        div_s = self.hutchinson_div_score(x, t, model)

        # computing the divergnce(v)
        div_v = -0.5 * beta_t * d - 0.5 * g2 * div_s

        return v, div_v