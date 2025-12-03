import torch
import numpy as np
from typing import Tuple, Optional
import math

class GaussianDiffusionSDE:
    def __init__(self, 
                 beta_min: float = 0.1, 
                 beta_max: float = 20, 
                 N: int = 1000, 
                 schedule: str = "linear",
                 sde_type: str = "subVP"):
        """
        Constructs a generalized SDE framework supporting both VP and subVP formulations.

        Args:
            beta_min: Value of beta(0).
            beta_max: Value of beta(1).
            N: Number of discretization steps.
            schedule: 'linear' or 'exponential' noise schedule.
            sde_type: 'VP' (Variance Preserving) or 'subVP' (Sub-Variance Preserving).
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.schedule = schedule
        
        if sde_type not in ('VP', 'subVP'):
            raise ValueError("sde_type must be 'VP' or 'subVP'")
        self.sde_type = sde_type

        # Validation for schedule
        if schedule not in ('linear', 'exponential'):
            raise ValueError("Schedule must be 'linear' or 'exponential'")
        
        if self.schedule == "exponential":
            self._k = float(torch.log(torch.tensor(self.beta_1 / self.beta_0)))

    # ----------------------------------------------------------------
    # Noise Schedule Definitions (Shared)
    # ----------------------------------------------------------------
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the noise rate beta(t)."""
        if self.schedule == "linear":
            return self.beta_0 + t * (self.beta_1 - self.beta_0)

        if self.schedule == "exponential":
            k = t.new_tensor(self._k)
            return t.new_tensor(self.beta_0) * torch.exp(k * t)
    
    def B(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the integral B(t) = ∫_0^t β(s) ds."""
        if self.schedule == "linear":
            return t * self.beta_0 + 0.5 * t**2 * (self.beta_1 - self.beta_0)
        
        if self.schedule == "exponential":
            k = t.new_tensor(self._k)
            beta0 = t.new_tensor(self.beta_0)
            return (beta0 / k) * (torch.exp(k * t) - 1)

    # ----------------------------------------------------------------
    # SDE Coefficients
    # ----------------------------------------------------------------
    def get_g_squared(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the diffusion coefficient squared g(t)^2.
        
        VP SDE:    g(t)^2 = β(t)
        subVP SDE: g(t)^2 = β(t) * [1 - exp(-2∫β(s)ds)]
        """
        beta_t = self.beta(t)
        
        if self.sde_type == "VP":
            return beta_t
            
        elif self.sde_type == "subVP":
            B_t = self.B(t)
            discount = 1.0 - torch.exp(-2.0 * B_t)
            return beta_t * discount
    
    def get_alpha_original(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes alpha(t) = 1 - (exp(-∫_0^t β(s) ds))
        """
        B_t = self.B(t)
        alpha_t = 1 - torch.exp(-B_t)
        return alpha_t # changed from squared

    def sde_coeff(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # previous subVP_sde function
        """
        Returns drift and diffusion coefficients evaluated at (x,t).
        
        Drift is f(x,t) = -1/2 * β(t) * x (Shared by both).
        Diffusion is g(t) (Dependent on type).
        """
        beta_t = self.beta(t)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(self.get_g_squared(t))
        return drift, diffusion

    # ----------------------------------------------------------------
    # Marginal Distributions (Transition Kernels)
    # ----------------------------------------------------------------
    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Computes mean coefficient: exp(-1/2 * B(t))"""
        return torch.exp(-0.5 * self.B(t))

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes marginal variance σ_t^2.
        
        VP:    σ_t^2 = 1 - exp(-B(t))
        subVP: σ_t^2 = (1 - exp(-B(t)))^2
        """
        B_t = self.B(t)
        term = 1 - torch.exp(-B_t)
        
        if self.sde_type == "VP":
            return term
        elif self.sde_type == "subVP":
            return term ** 2

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes p_0t(x(t)|x(0)).
        Returns (mean, std).
        """
        mean_c = self.mean_coeff(t)
        variance = self.var(t).clamp(min=1e-12)
        std = torch.sqrt(variance)
        
        mean = mean_c[:, None, None, None] * x0
        return mean, std

    # ----------------------------------------------------------------
    # Forward Perturbation
    # 1. Close form solution of a continuous SDE
    # 2. Actual step perturbation
    # ----------------------------------------------------------------
    def perturb_closed(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample X_t | X_0 using the closed-form transition kernel.
        Operation:
        1. Compute closed-form mean and std of X_t | X_0.
        2. Draw epsilon ~ N(0, I) with the same shape as x_0 if not provided.
        3. Return x_t = mean + std * epsilon, along with epsilon and std.

         Notes:
          - Fixed x_0, t, and noise.
          - Suitable for training score/ε-predictor networks with known std.
          
        Args:
        x_0: (B,C,H,W), t:(B,)
        """
        mean, std = self.marginal_prob(x_0, t)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = mean + std[:, None, None, None] * noise
        return x_t, noise, std

    def perturb_simulate_path(self, x_0: torch.Tensor, t_end: float = 1.0, steps: int = 500, seed: int = 42, eps: float = 1e-12):
        """Sample X_t by perturbing X_0 with gaussian noise at time t

        Operation:
        1. Compute simulate path for of X_t | X_t-1 and updating X_t values for steps time
        2. omputing the mean and std at time t
        3. Calculating the implied eps

         Notes:
          - With fixed x_0, t, and noise.
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
            drift, diffusion = self.sde_coeff(x, t_k)
            
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) # we generate Gaussian Noise, with same device and dtype as x
            x = x + drift * dt + diffusion[:, None, None, None] * (dt ** 0.5) * noise #sqrt(dt) is needed because it works as stabilizing term for the variance
        
        t_tensor = torch.full((cnt,), float(t_end), device = device, dtype = dtype)
        mean_t, std_t = self.marginal_prob(x_0, t_tensor)
        eps_implied = (x - mean_t) / (std_t[:, None, None, None] + 1e-12) #noise tensor
        return x, eps_implied, std_t

    # ----------------------------------------------------------------
    # Reverse Process
    # 1. Reverse with Euler step
    # 2. Reverse with ODE (probability flow)
    # ----------------------------------------------------------------
    
    def reverse_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float, scores: torch.Tensor, gen: torch.Generator = None) -> torch.Tensor:
        """
        Generic Euler - Marayuama method:
        
        dx = [ -1/2 β(t) x - g(t)^2 sθ(x,t) ] dt + g(t)sqrt(|dt|)z
        where z ~ N(0, I)

        Update:
        x <- x + dx
        t <- t + dt
        """
        beta_t = self.beta(t)
        g2 = self.get_g_squared(t)
        
        drift = (-0.5 * beta_t[:, None, None, None] * x) - (g2[:, None, None, None] * scores)
        
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        diffusion_step = torch.sqrt(g2 * abs(dt))[:, None, None, None] * noise
        
        return x + drift * dt + diffusion_step

    def probability_flow_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float, scores: torch.Tensor):
        """
        Deterministic PF-ODE with Euler step:
        dx = [ -1/2 β(t) x  - 1/2g(t)^2 sθ(x,t) ] dt
        """

        beta_t = self.beta(t)
        g2 = self.get_g_squared(t)
        drift = (-0.5 * beta_t[:, None, None, None] * x) - (0.5 * g2[:, None, None, None] * scores)
        x_ret = x + drift * dt
        return x_ret

    # ----------------------------------------------------------------
    # Reverse Correctors
    # 1. Langevin Corrector
    # ----------------------------------------------------------------

    @torch.no_grad()
    def corrector_langevin(self, x: torch.Tensor, t: torch.Tensor, scores: torch.Tensor, n_steps: int = 50, target_snr: float = 0.16, gen: torch.Generator = None, model: torch.nn.Module = None):
        """
        Corrector-Langevin
        x ← x + α s(x,t) + sqrt(2α) z, with α set to reach target SNR per batch.

        Details:
        Repeat for n_steps
            1. sample noise: z ~ N(0, I)
            2. compute norms ||\nabla log p|| and ||z||
            3. adapt the step size: α = 2(target_snr * ||z||/||\nabla log p||)^2
            4. update x value: x ← x + α s(x,t) + sqrt(2α) z
        """
        for _ in range(n_steps):
            
            if i > 0:
                _, std_t = sde.marginal_prob(x, t)
                eps_pred = model(x, t)
                scores = - eps_pred / (std_t[:, None, None, None] + 1e-12)
                
            noise = torch.randn(x.shape, device = x.device, dtype = x.dtype, generator = gen)
            # per-sample adaptive step size
            grad_norm = scores.flatten(1).norm(dim=1).clamp_min(1e-12)
            noise_norm = noise.flatten(1).norm(dim=1).clamp_min(1e-12)
            step_size = (target_snr * noise_norm / grad_norm) ** 2 *2.0
            # new x
            x = x + step_size[:, None, None, None] * scores + torch.sqrt(2.0 * step_size)[:, None, None, None] * noise
        
        return x
    # ------------------------------------------------------------------
    # Reverse Likelihood computation
    # 1. Compute the vector field representing the probability flow ODE
    # 2. Compute Log-density of a standard normal
    # 3. Hutchinson divergence score estimation
    # 4. Likelihood Euler Step
    # ------------------------------------------------------------------

    def v_field(self, x: torch.Tensor, t: torch.Tensor, scores: torch.Tensor):
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
        g2 = self.get_g_squared(t).view(-1, *([1] * (x.ndim -1)))
        return -0.5 * beta_t * x - 0.5 *g2 * scores

    def standard_normal_logprob(self, x: torch.Tensor):
        """
        Computing the log-density of a d-dimensional standard normal N(0,1) evaluated at x
        Args:
        - x in R^dWhy 
        - log N(0,I)(x) = -1/2 (∥x∥^2 + d log2π)
        """
        d = x[0].numel() # retrievs the flattened dimensionality per sample
        batch_size = x.size(0)
        quadratic = x.view(batch_size, -1).pow(2).sum(dim=1)
        return -0.5 * (quadratic + d * math.log(2 * math.pi))
    
    def hutchinson_div_score(self, x: torch.Tensor, t: torch.Tensor, scores: torch.Tensor, estimator = "rademacher"):
        """
        Estimate the diverge of the score: ∇⋅s_θ(x,t) = tr J(x), where J(x) = ∂s_𝜃/∂x.

        
        For Hutchunson's Identity: e ~ N(0,I): tr J = E_e[e^T J e], the trace of the Jacobian is equal to the expected value, over epsilon, of epsilon-transposed J epsilon.
        Args:
        - e ~ N(0,I)
        - ϕ(x)=⟨s_θ(x,t),e⟩ = ∑_i s_i(x,t)e_i
        - jte = ∇_x φ(x), which is J(x)^Te by the chain rule
        - then we can write ⟨(J^Te), e⟩ = e^TJe, which givs us an unbiased estimate of the trace of J
        """
        x_req = x.detach().requires_grad_(True)
        
        # Sample noise
        if estimator == "rademacher":
             e = (torch.randint_like(x_req, low=0, high=2).float() * 2.0 - 1.0)
        elif estimator == "gaussian":
             e = torch.randn_like(x_req)
        else:
            raise ValueError("Wrong estimator value for the Hutchunson's identity.")
        
        # Compute v(x) *inside* the graph
        _, std_t = self.marginal_prob(x_req, t)
        eps_pred = model(x_req, t)
        
        scores = - eps_pred / (std_t[:, None, None, None] + 1e-20)
        
        v_out = self.v_field(x_req, t, scores)

        # Vector-Jacobian Product (VJP)
        grad_v_e = torch.autograd.grad(
            outputs=(v_out * e).sum(), 
            inputs=x_req, 
            create_graph=False # We do not compute second order derivatives
        )[0]
        
        # Trace Estimate: e^T * (J^T * e) = e^T * grad_v_e
        div_v = (grad_v_e * e).flatten(1).sum(dim=1)
        
        return v_out.detach(), div_v

    def likelihood_euler_step(self, x, t, model, estimator):
        # Calculate v and div_v together
        v, div_v = self.hutchinson_div_v(x, t, model, estimator)
        
        return v, div_v
    
    # ----------------------------------------------------------------
    # DPM-Solver Utilities (Likelihood & SNR)
    # 1. Computing the log Signal-to-Noise Ratio (SNR)
    # 2. Compute the inverse of B(t)
    # 3. Compute the inverse of lambda
    # 4. DPM-Solver ++ update lambda
    # ----------------------------------------------------------------
    def get_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes log-SNR λ(t) = log( α_t / σ_t ).
        
        Where for sub-VP:
            α_t = exp(-0.5 * B(t))
            σ_t = 1 - exp(-B(t))
        
        Where for VP:
            α_t = exp(-0.5 * B(t))
            σ_t = sqrt(1 - exp(-B(t)))
        """
        alpha_t = self.mean_coeff(t)
        variance = self.var(t)
        sigma_t = torch.sqrt(variance)
        return torch.log(alpha_t / (sigma_t + 1e-12))

    def inverse_B(self, B_val: torch.Tensor) -> torch.Tensor:
        """
        Computes time t by inverting the noise schedule integral B(t).

        Solves for t:
            B(t) = ∫_0^t β(s) ds = B_val

        Handles both 'linear' (quadratic in t) and 'exponential' schedules.
        """
        if self.schedule == "linear":
            quadratic_num = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 + self.beta_0) * B_val)
            quadratic_den = self.beta_1 - self.beta_0
            return quadratic_num / quadratic_den

        elif self.schedule == "exponential":
            k = self._k
            term = (B_val * k / self.beta_0) + 1.0
            return torch.log(term) / k 
        
        raise ValueError("Invalid schedule")

    def inverse_lambda(self, lambda_val: torch.Tensor) -> torch.Tensor:
        """
        Computes time t from log-SNR λ.

        Method:
        1. Invert the Sub-VP relation for λ(B):
           λ = log( exp(-0.5 * B) / (1 - exp(-B)) )
           
        2. Solve for B:
           B = -2 * log( (-1 + √(1 + 4e^2λ)) / 2e^λ )
           
        3. Solve t = B⁻¹(B_val).
        """
        # Common term: e^λ
        exp_lambda = torch.exp(lambda_val)

        if self.sde_type == "subVP":
            # Relationship: e^λ = e^{-0.5 B} / (1 - e^{-B})
            y = (-1 + torch.sqrt(1 + 4 * exp_lambda ** 2)) / (2.0 * exp_lambda)
            B_val = -2.0 * torch.log(y)

        elif self.sde_type == "VP":
            # Relationship: e^λ = e^{-0.5 B} / sqrt(1 - e^{-B})
            # Implies: e^{2λ} = e^{-B} / (1 - e^{-B})
            # Let z = e^{-B}. Then e^{2λ} = z / (1 - z)
            # z = e^{2λ} / (1 + e^{2λ}) = 1 / (1 + e^{-2λ})
            # B = -log(z) = log(1 + e^{-2λ})
            B_val = torch.log(1.0 + torch.exp(-2.0 * lambda_val))
            # B_val = torch.nn.functional.softplus(-2.0 * lambda_val)

        return self.inverse_B(B_val)

    def dpm_solver_update_lambda(self, x, lam_curr, lam_next, eps_curr, prev_eps, prev_h, order):
        """
        Performs one DPM-Solver step from λ_t to λ_s (where s < t).

        Exact Solution:
            x_s = (α_s / α_t) * x_t - σ_s * ∫[λ_t to λ_s] e^(λ - λ_s) * ε_θ(x_λ, λ) dλ

        Implementation:
            Approximates the integral using Taylor expansion of ε_θ 
            (Order k = 1, 2, 3) based on history h.
        """
        h = lam_next - lam_curr
        phi_1 = torch.expm1(h)

        t_curr = self.inverse_lambda(lam_curr.unsqueeze(0))
        t_next = self.inverse_lambda(lam_next.unsqueeze(0))


        alpha_curr = self.mean_coeff(t_curr)
        alpha_next = self.mean_coeff(t_next)
        
        sigma_next = torch.sqrt(self.var(t_next))
        ratio = alpha_next / alpha_curr

        if order == 1 or len(prev_eps) == 0:
            integral = phi_1 * eps_curr
        
        elif order == 2 or len(prev_eps) == 1:
            eps_prev = prev_eps[-1]
            h_prev = prev_h[-1]

            r0 = h_prev / h

            D1 = (1.0 / r0) * (eps_curr - eps_prev)
            integral = phi_1 * eps_curr + (phi_1 - h) * D1
            
        else:
            eps_s1 = prev_eps[-1]
            eps_s2 = prev_eps[-2]

            h_s1 = prev_h[-1]
            h_s2 = prev_h[-2]

            r0 = h_s1 / h
            r1 = h_s2 / h

            D1_0 = (1.0 / r0) * (eps_curr - eps_s1)
            D1_1 = (1.0 / r1) * (eps_s1 - eps_s2)

            D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

            phi_2 = phi_1 - h
            phi_3 = phi_2 - 0.5 * (h ** 2)
            # we are taking Taylor overall: e^h - h - 0.5 * (h^2)

            integral = phi_1 * eps_curr + phi_2 * D1_0 + phi_3 * D2
            # so we are taking:
            # phi_1  * eps_curr = assuming noise is constat
            # phi_2 * D1_0: adjusting the path based on the slope
            # phi_3 * D2: adjusting the path based on the curvature

        x_next = ratio.to(x.device) * x - sigma_next.to(device) * integral
        return x_next