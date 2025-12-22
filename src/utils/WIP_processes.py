import torch
import torch.nn as nn

import time
import torchvision.utils
import matplotlib.pyplot as plt

from src.utils.WIP_SDE import SDE, BetaScheduleSDE, SubVPSDE, VESDE, VPSDE


def _expand_batch_vector_to(x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Expand a (B,) vector to match the shape of x (B, C, H, W, ...).

    Args:
        x: tensor of shape (B, ...)
        vec: tensor of shape (B,)

    Returns:
        tensor of shape (B, 1, 1, 1, ...) broadcastable with x
    """
    while vec.dim() < x.dim():
        vec = vec.unsqueeze(-1)
    return vec


class Diffusion_Processes:
    def __init__(self, cfg: dict):
        self.N = cfg["N"]
        self.sde_type = cfg["sde_type"].lower()
        self.cfg = cfg

        if self.sde_type == "ve":
            # You can pass sigma_min, sigma_max, etc. via cfg if you want
            self.sde: SDE = VESDE(N=self.N)
        elif self.sde_type == "vp":
            # Default to sub-VP; you can also pass beta_min, beta_max, schedule, etc. via cfg
            self.sde: SDE = VPSDE(N=self.N)
        else:
            self.sde: SDE = SubVPSDE(N=self.N)

    @torch.no_grad()
    def forward_process(self, z0: torch.Tensor, t: torch.Tensor = None):
        """
        Forward diffusion: add noise to clean data z0 according to the chosen SDE.

        This uses the closed-form marginal p_t(z | z0):

            z_t = mean(z0, t) + std(t) * eps,  eps ~ N(0, I)

        Args:
            z0: clean data, shape (B, C, H, W) or similar.

        Returns:
            z_t: noised data at random time t, same shape as z0
            t:  time vector, shape (B,)
            eps: the Gaussian noise used, same shape as z0
        """
        device = z0.device
        B = z0.size(0)

        if t == None:
            # Sample a time for each example: t ~ Uniform(0, T)
            t = torch.rand(B, device=device) * self.sde.T

        # Get closed-form mean and std of p_t(z | z0)
        mean, std = self.sde.marginal_prob(z0, t)  # mean: (B, ...), std: (B,)

        # Sample noise
        eps = torch.randn_like(z0)

        # Broadcast std to match z0

        # Construct z_t
        z_t = mean + std[:, None, None, None] * eps

        return z_t, t, eps

    @torch.no_grad()
    def reverse_process(
        self,
        model: nn.Module,
        shape,
        num_steps: int = None,
        probability_flow: bool = False,
        device: torch.device = None,
        y: torch.Tensor = None #CFG conditioning
    ):
        """
        Reverse diffusion: sample from the data distribution using the learned model.

        This integrates the reverse-time SDE/ODE defined by self.sde.reverse().

        Assumptions:
            - model(x, t) returns the score ∇_x log p_t(x) (Song-style score model).
              If your model predicts noise ε instead, you must wrap it and convert
              to a score before passing it here.

        Args:
            model: neural net implementing score(x, t).
            shape: shape of the samples to generate, e.g. (B, C, H, W).
            num_steps: number of reverse-time discretization steps (default: self.N).
            probability_flow: if True, use probability flow ODE (deterministic);
                              if False, use reverse SDE (stochastic).

        Returns:
            x: generated samples, tensor of shape `shape`.
        """
        if num_steps is None:
            num_steps = self.N

        # # --- FIX: Check if model is a function or a class ---
        # if device is None:
        #     if hasattr(model, "parameters"):
        #         # It's a real PyTorch model
        #         device = next(model.parameters()).device
        #     else:
        #         # It's a function (wrapper), so we assume CUDA or CPU
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = "cpu" #next(model.parameters()).device
        B = shape[0]
        T = self.sde.T
        # null_y = torch.full((B,), self.num_classes, device = device) # index for null token/unconditioning
        print(f"This is our SDE: {self.sde}")
        print(f"This is the value of T: {T}")

        # # Define the score function using the model.
        # def score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        #     # # t is (B,) – pass as is; adapt if your model expects a different format.
        #     # return model(x, t)
        #     model_out = model(x, t)
        #     _, std = self.sde.marginal_prob(x, t)  # std is shape (B,)
            
        #     #    We view std as (B, 1, 1, 1)
        #     std = std.view(*std.shape, *([1] * (x.dim() - 1)))
        #     if self.sde_type == "ve":
        #         score = model_out / (std + 1e-6)
        #     else:
        #         score = - model_out / (std + 1e-6)
        #     return score
        def score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            Computes the score using the pre-trained model.
            Handles the mapping from continuous SDE time t to model-specific inputs.
            """
            # 1. Get the marginal std (sigma) from the SDE
            #    std shape: (B,)
            _, std = self.sde.marginal_prob(x, t)

            # 2. Prepare inputs for the specific model type
            if self.sde_type == "ve":
                # VE Models (NCSN++) expect the actual SIGMA value as input
                # diffusers implementation of NCSN++ takes continuous sigmas
                model_input_t = t
            else:
                # VP Models (DDPM) expect discrete indices [0, 999]
                # Map continuous t \in [0, 1] to discrete steps
                # We clamp to ensure we don't hit 1000 which is out of bounds
                model_input_t = (t * 999).long().clamp(max=999)

            # 3. Forward Pass
            # .sample is REQUIRED because diffusers models return an output object
            model_out = model(x, model_input_t)

            # 4. Convert Output to Score
            # Reshape std for broadcasting: (B, 1, 1, 1)
            std = std.view(*std.shape, *([1] * (x.dim() - 1)))
            
            if self.sde_type == "ve":
                # VE: Model predicts score * sigma (approx).
                # score = output / sigma
                score = model_out / (std + 1e-6)
            else:
                # VP: Model predicts noise (epsilon).
                # score = -epsilon / sigma
                score = -model_out / (std + 1e-6)

            return score
        # ---------------- CFG SCORE ----------------
        # def cfg_score_fn(x: torch.Tensor,t: torch.Tensor):
        #     x_combined = torch.cat([x,x], dim=0)
        #     t_combined = torch.cat([t,t], dim=0)
        #     y_combined = torch.cat([y, null_y], dim=0)
            
        #     # Getting predictions
        #     model_out = model(x_combined, t_combined, y_combined)
        #     eps_cond, eps_uncond = model_out.chunk(2, dim=0)
            
        #     # CFG Extrapolation
        #     eps_cfg = eps_uncond + self.cfg['guidance_scale'] * (eps_cond - eps_uncond)
        #     # Convert to Score
        #     _, std = self.sde.marginal_prob(x,t)
        #     std = std.view(*std.shape, *([1] * (x.dim() - 1)))

        #     if self.sde_type == 've':
        #         return eps_cfg / (std + 1e-6)
        #     else:
        #         return -eps_cfg/(std + 1e-6)

        # Build reverse-time SDE/ODE
        # if cfg['cfg']:
        # rdse: SDE = self.sde.reverse(cfg_score_fn, probability_flow=probability_flow)
        rsde: SDE = self.sde.reverse(score_fn, probability_flow=probability_flow)

        # Initialize from the prior at time T
        x = self.sde.prior_sampling(shape).to(device)
        print(f"Check prior {self.sde_type}: Mean = {x.mean()}, Std = {x.std()}")

        k = max(num_steps//10, 1)
        start_time = time.time()
        
        # Time discretization from T -> 0
        for i in reversed(range(3,num_steps)):
            t_i = torch.full((B,), T * i / num_steps, device=device)
            f, G = rsde.discretize(x, t_i)  # f: (B, ...), G: (B,)

            G_b = _expand_batch_vector_to(x, G)

            if probability_flow or i == 0:
                noise = 0.0
            else:
                noise = torch.randn_like(x)

            x_prev = x
            x = x - f + G_b * noise

            # ---- log stats + show images every 10% of steps ----
            if (i % k == 0) or (i < 15):
                print("Statistics for each timestep:")
                print(f"Drift: mean drift {f.mean()}, std drift: {f.std()}")
                print(f"Diffusion: mean diff {G_b.mean()}, std diff: {G_b.std()}")
                print(f"Update: mean difference (x_new - x_prev): mean_difference = {(x-x_prev).mean()}, std_difference = {(x-x_prev).std()}")
                
                x_cpu = x.detach().cpu()  # (B, C, H, W)
                B, C, H, W = x_cpu.shape
            
                # Global statistics over the whole tensor
                mean = x_cpu.mean().item()
                std = x_cpu.std().item()
                x_min = x_cpu.min().item()
                x_max = x_cpu.max().item()
            
                # Per-channel statistics: reduce over batch + spatial dims, keep channel dim
                # x_cpu.mean(dim=(0,2,3)) -> (C,)
                ch_means = x_cpu.mean(dim=(0, 2, 3))
                ch_stds  = x_cpu.std(dim=(0, 2, 3))
            
                # For min/max, easiest is to flatten spatial dims and reduce:
                flat = x_cpu.view(B, C, -1)            # (B, C, H*W)
                ch_mins = flat.min(dim=-1).values.mean(dim=0)  # (C,) average over batch mins
                ch_maxs = flat.max(dim=-1).values.mean(dim=0)  # (C,) average over batch maxs
            
                # Timing info
                elapsed_time, start_time = time.time() - start_time, time.time()
            
                print(
                    f"[reverse step {i+1}/{num_steps} | i={i} | t={t_i[0].item():.4f}]\n"
                    f"global: mean={mean:.4f}, std={std:.4f}, min={x_min:.4f}, max={x_max:.4f}"
                )
                for c in range(C):
                    print(
                        f"  ch {c}: mean={ch_means[c]:.4f}, std={ch_stds[c]:.4f}, "
                        f"min≈{ch_mins[c]:.4f}, max≈{ch_maxs[c]:.4f}"
                    )
                print(
                    f"Time of last {k} steps: {elapsed_time:.3f}. "
                    f"Time remaining (rough heuristic): {(k - i//k) * elapsed_time:.3f}.\n"
                )
            
                # visualize a few samples
                x_vis = x_cpu.clamp(-1.0, 1.0)
                x_vis = (x_vis + 1.0) / 2.0
            
                grid = torchvision.utils.make_grid(x_vis[:16], nrow=4)
                grid = grid.permute(1, 2, 0).numpy()
            
                plt.figure(figsize=(4, 4))
                plt.imshow(grid)
                plt.title(f"Reverse step {i+1}/{num_steps}")
                plt.axis("off")
                plt.tight_layout()
                plt.show()

        return x
