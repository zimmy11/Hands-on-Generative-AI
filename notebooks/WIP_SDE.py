import abc
import math
from typing import Tuple

import numpy as np
import torch


# ============================================================
# Abstract base SDE, following Yang Song's design
# ============================================================

class SDE(abc.ABC):
    """
    Abstract SDE class. All concrete SDEs (VE, VP, subVP) should subclass this.
    Methods are designed for mini-batches of inputs.
    """

    def __init__(self, N: int = 1000):
        """
        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE (typically 1.0)."""
        pass

    @abc.abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Instantaneous drift and diffusion of the forward SDE at (x, t).

        Returns:
            drift: Tensor with same shape as x
            diffusion: Tensor of shape (B,) or broadcastable to x
        """
        pass

    @abc.abstractmethod
    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters of the marginal p_t(x | x0).

        Returns:
            mean: Tensor with same shape as x0
            std:  Tensor of shape (B,) or broadcastable to x0
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape) -> torch.Tensor:
        """
        Sample from the prior distribution p_T(x).
        """
        pass

    @abc.abstractmethod
    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-density of the prior p_T(z).

        Args:
            z: latent tensor of shape (B, ...)

        Returns:
            logp: shape (B,)
        """
        pass

    def discretize(self, x: torch.Tensor, t: torch.Tensor):
        """
        Euler–Maruyama discretization:

            x_{i+1} = x_i + f_i(x_i) * dt + G_i * sqrt(dt) * z

        where f_i and G_i come from self.sde(x, t).

        Args:
            x: state, shape (B, C, H, W) or similar
            t: time, shape (B,)

        Returns:
            f: drift * dt, same shape as x
            G: diffusion * sqrt(dt), shape (B,)
        """
        dt = self.T / self.N
        drift, diffusion = self.sde(x, t)
        # diffusion is assumed (B,) or broadcastable, keep Song's pattern
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device, dtype=t.dtype))
        return f, G

    def reverse(self, score_fn, probability_flow: bool = False):
        """
        Construct the reverse-time SDE/ODE as a new SDE instance.

        Args:
            score_fn: function(x, t) -> score, same shape as x
            probability_flow: if True, build the probability flow ODE (diffusion=0);
                              otherwise build the reverse-time SDE.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        parent_cls = self.__class__

        class RSDE(parent_cls):
            def __init__(self):
                # We intentionally do NOT re-run parent's __init__ with extra params.
                # We only keep N and mark this as a reverse process wrapper.
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                # Assume diffusion is (B,)
                diff_sq = diffusion[:, None, None, None] ** 2
                factor = 0.5 if self.probability_flow else 1.0
                drift = drift - diff_sq * score * factor
                diffusion_rev = 0.0 if self.probability_flow else diffusion
                return drift, diffusion_rev

            def discretize(self, x, t):
                f, G = discretize_fn(x, t)
                score = score_fn(x, t)
                factor = 0.5 if self.probability_flow else 1.0
                G_sq = G[:, None, None, None] ** 2
                rev_f = f - G_sq * score * factor
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


# ============================================================
# Beta-schedule base class for VP / sub-VP family
# ============================================================

class BetaScheduleSDE(SDE):
    """
    Base class for SDEs driven by a noise schedule β(t) on [0, 1].
    This is meant to be shared by VP and sub-VP variants.

    It provides:
      - beta(t): instantaneous β(t)
      - B(t): ∫_0^t β(s) ds
      - mean_coeff(t) = exp(-0.5 * B(t))
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        schedule: str = "linear",
    ):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        if schedule not in ("linear", "exponential"):
            raise ValueError("schedule must be 'linear' or 'exponential'")
        self.schedule = schedule

        if self.schedule == "exponential":
            # k = log(beta_1 / beta_0)
            self._k = float(torch.log(torch.tensor(self.beta_1 / self.beta_0)))

    @property
    def T(self) -> float:
        return 1.0

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        β(t): either linear or exponential between beta_0 and beta_1.
        """
        if self.schedule == "linear":
            return self.beta_0 + t * (self.beta_1 - self.beta_0)

        # exponential
        k = t.new_tensor(self._k)
        return t.new_tensor(self.beta_0) * torch.exp(k * t)

    def B(self, t: torch.Tensor) -> torch.Tensor:
        """
        B(t) = ∫_0^t β(s) ds for the chosen schedule.
        """
        if self.schedule == "linear":
            return t * self.beta_0 + 0.5 * t**2 * (self.beta_1 - self.beta_0)

        if self.schedule == "exponential":
            k = t.new_tensor(self._k)
            beta0 = t.new_tensor(self.beta_0)
            return (beta0 / k) * (torch.exp(k * t) - 1.0)

        # Should never reach here
        raise ValueError("Invalid schedule in BetaScheduleSDE")

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """
        α(t) = exp(-0.5 * B(t)). This is shared by VP and sub-VP.
        """
        return torch.exp(-0.5 * self.B(t))

    # var(t) is model-dependent:
    #   - VP:   var(t) = 1 - exp(-B(t))
    #   - subVP:var(t) = (1 - exp(-B(t)))^2
    # so we leave var() and marginal_prob() to subclasses.


# ============================================================
# sub-VP SDE in unified interface
# ============================================================

class SubVPSDE(BetaScheduleSDE):
    """
    Sub-Variance-Preserving SDE that is convenient for likelihoods.
    Implements Yang Song-style interface using the shared beta schedule.
    """

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """
        For sub-VP, the effective noise scale σ(t) is:
            σ(t) = 1 - exp(-B(t))
        and the variance is σ(t)^2.
        """
        s = 1.0 - torch.exp(-self.B(t))
        return s**2

    def g_squared(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sub-VP diffusion coefficient:
            g(t)^2 = β(t) [ 1 - exp(-2 * B(t)) ]
        """
        beta_t = self.beta(t)
        B_t = self.B(t)
        discount = 1.0 - torch.exp(-2.0 * B_t)
        return beta_t * discount

    # ---- Core SDE interface ----

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        """
        Instantaneous SDE coefficients for sub-VP:

            dX_t = f(X_t, t) dt + g(t) dW_t

        with
            f(x, t) = -1/2 β(t) x
            g(t)^2  = β(t)[1 - exp(-2∫_0^t β(s) ds)]
        """
        beta_t = self.beta(t)  # (B,)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(self.g_squared(t))  # (B,)
        return drift, diffusion

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form marginal of X_t | X_0 for sub-VP:

            mean = α(t) * x0,  where α(t) = exp(-0.5 * B(t))
            std  = sqrt( var(t) ), where var(t) = (1 - exp(-B(t)))^2
        """
        alpha_t = self.mean_coeff(t)            # (B,)
        std = torch.sqrt(self.var(t))          # (B,)
        mean = alpha_t[:, None, None, None] * x0
        return mean, std

    def prior_sampling(self, shape):
        """
        Standard normal prior N(0, I) at time T.
        """
        return torch.randn(*shape)

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-density of standard normal N(0, I) evaluated at z.
        """
        batch_size = z.size(0)
        d = z[0].numel()
        quadratic = z.view(batch_size, -1).pow(2).sum(dim=1)
        return -0.5 * (quadratic + d * math.log(2.0 * math.pi))


# ============================================================
# Variance Exploding SDE in unified interface
# ============================================================

class VESDE(SDE):
    """
    Variance Exploding SDE, re-written as a subclass of SDE.
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, N: int = 1000):
        """
        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps.
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        # Precomputed discrete sigmas for SMLD-style discretization
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )

    @property
    def T(self) -> float:
        return 1.0

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time-dependent noise scale σ(t) for VE:
            σ(t) = σ_min (σ_max / σ_min)^t
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion coefficient g(t) of the VE SDE:
            g(t) = σ(t) * sqrt( 2 (log σ_max - log σ_min) )
        """
        sigma = self.sigma_t(t)
        return sigma * torch.sqrt(
            torch.tensor(
                2.0 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                device=t.device,
                dtype=t.dtype,
            )
        )

    # ---- Core SDE interface ----

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        """
        VE SDE:

            dX_t = g(t) dW_t

        (zero drift; purely exploding variance).
        """
        drift = torch.zeros_like(x)
        diffusion = self.diffusion_coeff(t)  # (B,)
        return drift, diffusion

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form marginal of X_t | X_0 for VE:

            mean = x0
            std  = σ(t)
        """
        std = self.sigma_t(t)
        mean = x0
        return mean, std

    def prior_sampling(self, shape):
        """
        Gaussian prior with variance σ_max^2.
        """
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-density of N(0, σ_max^2 I).
        """
        batch_size = z.size(0)
        d = z[0].numel()
        quadratic = z.view(batch_size, -1).pow(2).sum(dim=1)
        return -0.5 * (
            quadratic / (self.sigma_max**2) + d * math.log(2.0 * math.pi * self.sigma_max**2)
        )

    # Optional: keep specialized SMLD discretization
    def discretize(self, x: torch.Tensor, t: torch.Tensor):
        """
        SMLD (NCSN) discretization for VE:

            x_{i+1} = x_i + G_i z_i

        where G_i = sqrt(σ_i^2 - σ_{i-1}^2).
        """
        timestep = (t * (self.N - 1) / self.T).long()
        sigmas = self.discrete_sigmas.to(t.device)
        sigma = sigmas[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            sigmas[timestep - 1],
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G
