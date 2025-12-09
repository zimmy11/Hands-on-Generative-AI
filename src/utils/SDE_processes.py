# previous name subVP_processes

from typing import Optional, Tuple
import torch
from src.utils.SDE import VESDE
import torch.nn as nn
from torchvision.utils import save_image
from src.utils.vae_utils import get_vae_encoder_func
import time
import os

class DiffusionProcesses:
    def __init__(self, configurations: dict):
        cfg = configurations['ForwardConfig']
        self.beta_min = cfg['beta_min']
        self.beta_max = cfg['beta_max']
        self.N = cfg['N']
        self.schedule = cfg['schedule']

    @torch.no_grad()
    def get_noised_latents(self, z0: torch.Tensor, configurations: dict, is_times: torch.Tensor = None):
        cfg = configurations['ForwardConfig']
        
        print(f"Forward till final: {cfg['final']}")
        t_tensor = is_times

        print(f"[DEBUG DiffusionProcesses] Using t values: Min={t_tensor.min().item():.4f}, Max={t_tensor.max().item():.4f}, Shape={t_tensor.shape}")
        sde = VESDE()

        z_t, epsilon, std = sde.perturb_closed(z0, t_tensor)
        
        return z_t, epsilon, std, sde

    @torch.no_grad()
    def run_forward(self, z0, configurations = dict(), is_times: torch.Tensor = None):
        """
        Execute the forward process for the latent noised variables with the parameters passed in Configurations
        """
        cfg = configurations
        
        z_t, epsilon, std, sde = self.get_noised_latents(
            z0, cfg, is_times = is_times)

        # To check actual noising we compute some core statistics
        z0_mean = z0.mean()
        z0_std = z0.std(unbiased = False)
        z0_standardized = (z0 - z0_mean)/z0_std
        z0_skew = torch.mean(z0_standardized ** 3)
        z0_kurtosis = torch.mean(z0_standardized ** 4)

        z_t_mean = z_t.mean()
        z_t_std = z_t.std(unbiased = False)
        z_t_standardized = (z_t - z_t_mean)/z_t_std
        z_t_skew = torch.mean(z_t_standardized ** 3)
        z_t_kurtosis = torch.mean(z_t_standardized ** 4)        
        
        print("Pre and Post noised values")
        print(f"Key statistics: min var = {sde.sigma_min}, max var = {sde.sigma_max}, n. steps = {sde.N}")
        print(f"Encoded images statistics: mean = {z0_mean}, std = {z0_std}, skew = {z0_skew}, kurtosis = {z0_kurtosis}")
        print(f"Noised Encoded images statistics: mean = {z_t_mean}, std = {z_t_std}, skew = {z_t_skew}, kurtosis = {z_t_kurtosis}")

        return z_t, epsilon, std, sde
    
    def sample_reverse(self, configurations: dict, model: nn.Module, save_dir: str = "./samples"):
        """
        We are implementing the sampling through reversing the SDE.

        Args:
        - cfg: define the configuration parameters of the reverse process
        - x are sampled form N(0, I), since the forward process brought us to the prior π(x)
        - dt: is a negative timestep T -> 0, where t0 = 1 (starting time) and t1 = 0 (ending time)

        Formulation:
        At every moment p_0t(x_t|x_0) = N(x_t; μ, σ^2I):
        1. log p(x_t) ∝ ||x_t -μ||^2/2σ^2_t
        2. \nabla log p(x_t) = - (x_t -μ)/σ^2_t
        3. Since: x_t = μ + σ_t eps -> x_t -μ = σ_t eps
        4. \nabla log p(x_t) = - eps / σ_t
        """
        cfg = configurations['ReverseConfig']
        device = cfg['device']
        vae_scale_factor = cfg.get('vae_scale_factor', 0.18215)
        _, decode_func = get_vae_encoder_func(device) 
        dtype = torch.float32

        sde = VESDE()
        
        gen = torch.Generator(device=cfg['device']).manual_seed(cfg['seed'])

        # CURRENT FOR VE: added sde.sigma_max at the end
        x = torch.randn(*cfg['shape'], device=cfg['device'], dtype=cfg['dtype'], generator=gen) * sde.sigma_max
        # PREVIOSU: x = torch.randn(*cfg['shape'], device = cfg['device'], dtype = cfg['dtype'], generator = gen)

        #Time discretization for reversion execution
        t_grid = torch.linspace(cfg['t0'] + cfg['eps'] , cfg['t1'] - cfg['eps'], cfg['N'] + 1, device = device, dtype = dtype)

        model = model.to(device = device, dtype = dtype).eval()
        
        start_time_fixed = time.time()
        start_time = time.time()
        n_steps = cfg['N']//10
        
        #Reverse process loop
        with torch.no_grad():
            for k in range(cfg['N']):

                if k % 100 == 0:
                    x_min, x_max = x.min().item(), x.max().item()
                    print(f"  Step {k}/{cfg['N']} - x range: [{x_min:.2f}, {x_max:.2f}]")
                    # if x_max > 100 or x_min < -100:
                    #     print("  WARNING: Latents exploding! Adding clamp.")
                    #     x = torch.clamp(x, -5.0, 5.0) # Safety clamp
                    if torch.isnan(x).any():
                        print("  CRITICAL: NaN in sampling loop!")
                        break
                    time_elapsed, start_time = time.time() - start_time, time.time()

                    print(f"Summary stats:\nSteps done: {k}\nTime of last {n_steps} steps: {time_elapsed}\nAverage time of last {n_steps} steps: {time_elapsed/n_steps}\nOverall time:{time.time()-start_time_fixed}")
                
                if k % 200 == 0 and k > 0 :
                    print("Generating intermediate samples...")
                    latents_to_decode = x / vae_scale_factor
                    with torch.no_grad():
                        x_decoded = decode_func(latents_to_decode)
                    os.makedirs(save_dir, exist_ok=True)
                    filename = f"LDM_{cfg['epochs']}_step_{k:04d}.png"
                    save_image(x_decoded, os.path.join(save_dir, filename), normalize=True, value_range=(-1, 1), nrow=4)
                    torch.cuda.empty_cache()
                    print(f"✅ Saved sample in: {os.path.join(save_dir, filename)}")



                t_k = t_grid[k].expand(cfg['shape'][0])
                t_k1 = t_grid[k+1].expand(cfg['shape'][0])
                dt = (t_k1[0] - t_k[0])

                # Extracting current standard deviation
                _, std_t = sde.marginal_prob(x, t_k)

                # Converting eps_pred (noise) into scores \nabla_x log p_t(x)
                eps_pred = model(x, t_k)
                scores = - eps_pred / (std_t[:, None, None, None] + 1e-12)
                

                

                #Predictor
                # if cfg['rev_type'] == "sde":
                x = sde.reverse_euler_step(x, t_k, dt, scores, gen = gen)
                # elif cfg['rev_type'] == "ode":
                #     x = sde.probability_flow_euler_step(x, t_k, dt, scores, gen = gen)
                
                # #Corrector
                # if cfg['corrector'] == True:
                #     x = sde.corrector_langevin(x, t_k1, scores, n_steps = cfg['n_corr'], target_snr = cfg['target_snr'], gen = gen, model = model)
            
        return x

        
    def run_reverse(self, model:nn.Module, likelihood: bool = False, configurations=  dict()):
        if not likelihood:
            return self.sample_reverse(configurations, model)
        else:
            # lcfg = LikelihoodConfig()
            # return self.log.likelihood_subvp_ode(
            raise ValueError("Attention Likelihood is still in validation phase. not available yet")
