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
        device: torch.device = None
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

        #next(model.parameters()).device
        B = shape[0]
        T = self.sde.T
        print(f"This is our SDE: {self.sde}")
        print(f"This is the value of T: {T}")


        def score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            Computes the score using the pre-trained model.
            Handles the mapping from continuous SDE time t to model-specific inputs.
            """
            # 1. Get the marginal std (sigma) from the SDE
            #    std shape: (B,)
            _, std = self.sde.marginal_prob(x, t)
            model_input_t = t

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

        # Build reverse-time SDE/ODE
        rsde: SDE = self.sde.reverse(score_fn, probability_flow=probability_flow)

        # Initialize from the prior at time T
        x = self.sde.prior_sampling(shape).to(device)
        print(f"Check prior {self.sde_type}: Mean = {x.mean()}, Std = {x.std()}")

        k = max(num_steps//10, 1)
        start_time = time.time()
        
        # Time discretization from T -> 0
        for i in reversed(range(num_steps)):
            t_i = torch.full((B,), T * i / num_steps, device=device)
            f, G = rsde.discretize(x, t_i)  # f: (B, ...), G: (B,)

            G_b = _expand_batch_vector_to(x, G)

            if probability_flow or i == 0:
                noise = 0.0
            else:
                noise = torch.randn_like(x)

            x = x - f + G_b * noise


            # ---- log stats + show images every 10% of steps ----
            if (i % k == 0) or (i < 15):
                x_cpu = x.detach().cpu()
        
                # discrete statistics
                mean = x_cpu.mean().item()
                std = x_cpu.std().item()
                x_min = x_cpu.min().item()
                x_max = x_cpu.max().item()

                elapsed_time, start_time = time.time() - start_time, time.time()
                print(
                    f"[reverse step {i+1}/{num_steps} | i={i} | t={t_i[0].item():.4f}]\n"
                    f"mean={mean:.4f}, std={std:.4f}, min={x_min:.4f}, max={x_max:.4f}\n"
                    f"Time of last {k} steps: {elapsed_time}. Time remaining {(k - i//10) * elapsed_time}.\n"
                )
                
                # visualize a few samples
                # if model works in [-1, 1], map to [0, 1] for display
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