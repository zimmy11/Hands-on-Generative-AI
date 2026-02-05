import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import wandb
import os
from tqdm import tqdm
import torchvision
from src.utils.vae_utils import get_vae_encoder_func
# Import dei tuoi moduli
from src.utils.sde_utils import Diffusion_Processes, SubVPSDE, VESDE, VPSDE
from src.utils.utils import setup
from src.utils.utils import log_denoising_step_wandb # La funzione creata sopra
from src.models.UNet import UNet
from src.models.components import EMAModel

# ---- LOADING THE MODEL ----------------------------------------
def _extract_state_dict(ckpt):
    for k in ("state_dict", "model",  "model_state_dict"): #, "ema"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt

def _remove_ema(sd: dict) -> dict:
    return {
        k: v for k, v in sd.items()
        if not (
            k.startswith("ema.") or
            k.startswith("ema_model.") or
            k.startswith("model_ema.")
        )
    }


def _strip_prefixes(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd):
        sd = {k.split("module.", 1)[1] if k.startswith("module.") else k: v for k, v in sd.items()}
    # if all(k.startswith("unet.") for k in sd):
    sd = {k.replace("unet.", "", 1) if k.startswith("unet.") else k: v for k, v in sd.items()}
    return sd

def load_model_from_local(model: nn.Module, ckpt_path: str, device: torch.device) -> bool:
    """Load checkpoint from local filesystem instead of Google Drive."""
    if not os.path.isfile(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return False
    print(f"Loading model weights from {ckpt_path} ...")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)#, safe_globals=[EMAModel])
    except Exception as e:
        print(f"[torch.load] failed: {e}")
        return False

    sd = _extract_state_dict(ckpt)
    sd = _strip_prefixes(sd)
    sd = _remove_ema(sd)



    num_tensors = len(sd)
    num_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(f"Checkpoint state_dict: {num_tensors} tensors, {num_params:,} total parameters")

    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}

    dropped = sorted(k for k in sd.keys() if k not in model_keys)
    if dropped:
        print(f"Dropping {len(dropped)} unexpected keys (showing up to 8):")
        for k in dropped[:]:
            print("  ", k)
        if len(dropped) > 8:
            print("  ...")

    res = model.load_state_dict(filtered, strict=False)
    print(f"Loaded. missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")

    model.to(device).eval()
    return True

# ---------------------------------------------

def t_to_index(t: torch.Tensor, T: float, N: int) -> torch.Tensor:
    # Map t in [0,T] -> idx in [0, N-1]
    idx = (t / T * (N - 1)).long()
    return idx.clamp(0, N - 1)

# def avg_model_error(model, loader, forward_process, device, vae_encoder, vae_scale_factor):
#     """
#     Simply compute the MSE between the prediction and the actual noise
#     """
#     print(">>> Calcolo NLL (Test Loss)...")
#     total_loss = 0
#     num_batches = 0
#     loss_fn = nn.MSELoss() # O la loss specifica usata nel training (weighted)

#     with torch.no_grad():
#         for batch in tqdm(loader, desc="NLL Computation"):
#             if isinstance(batch, list):
#                 x = batch[0].to(device)
#                 y = batch[1].to(device)
#             else:
#                 x = batch.to(device)
#                 y = None

#             latents = vae_encoder(x) 
#             latents = latents * vae_scale_factor # Importante: scalare i latenti!
            
#             # Ora usiamo i LATENTI per il forward process
#             t = torch.rand(latents.shape[0], device=device) * forward_process.sde.T
#             z_t, t, eps = forward_process.forward_process(latents, t)
            
#             # Prediction
#             # NOTA: Assicurati che score_fn sia coerente con come calcoli la loss in training
#             # Qui assumiamo output = eps prediction (standard DDPM/VP)
#             if forward_process.conditional and y is not None:
#                 pred = model(z_t, t, y.float())
#             else:
#                 null_y = torch.zeros((x.shape[0], 40), device=device)
#                 pred = model(z_t, t, null_y)
                
#             loss = loss_fn(pred, eps) # Semplificato
#             total_loss += loss.item()
#             num_batches += 1
            
#     return total_loss / num_batches


# -------------------- ODE EXACT COMPUTATION ------------------
import math
from torch.autograd import grad

def calculate_nll_ode(
    model,
    loader,
    diff_proc,
    device,
    vae_encoder,
    vae_scale_factor,
    n_steps=1000,
    eps_time=1e-5,
    hutchinson_samples=1,   # >1 reduces variance
    max_images = 0
):
    """
    Numerical estimate of log p(z0) via probability flow ODE:

        log p0(z0) = log pT(zT) + ∫_t0^T div(f_ode(z,t)) dt

    Returns average NLL (nats) over loader, in *latent space*.

    Assumptions:
      - model predicts epsilon under the perturbation kernel:
            z_t = mean(t)*z0 + std(t)*eps
        so score(z_t,t) = ∇_{z_t} log p_t(z_t) ≈ -eps_pred / std(t)
      - Uses diff_proc.sde.{sde, marginal_prob, prior_logp}
    """

    model.eval()

    # Disable grads for model params (we only need grads wrt state z for divergence)
    # by disabling parameters we are reducing memory, avoid builidng huge computation graphs for theta
    prev_req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    total_nll = 0.0
    total_count = 0
    sumsq_nll = 0.0

    T = float(diff_proc.sde.T)
    t0 = float(eps_time)
    ts = torch.linspace(t0, T, n_steps, device=device)
    dts = ts[1:] - ts[:-1]

    def _broadcast_like(vec_1d, x):
        # vec_1d: (B,), returns (B,1,1,1,...) to match x dims, useful for SDE functions that return scalar that needs to be multiplied by tensors
        return vec_1d.view(vec_1d.shape[0], *([1] * (x.dim() - 1)))

    for batch in tqdm(loader, desc="NLL (prob-flow ODE)"):
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
            y = batch[1].to(device) if len(batch) > 1 else None
        else:
            x = batch.to(device)
            y = None

        with torch.no_grad():
            z0 = vae_encoder(x) * vae_scale_factor

        # Initializing the ODE state and log-density correction accumulator
        z = z0.detach()
        B = z.shape[0]
        logp_correction = torch.zeros(B, device=device)

        # Prepare labels (keep your existing convention)
        if getattr(diff_proc, "conditional", False) and (y is not None):
            labels = y.float()
        else:
            # WARNING: adjust 40 to your actual label dim if your model expects labels always.
            labels = torch.zeros((B, 40), device=device)

        for i in range(n_steps - 1):
            t = ts[i]
            dt = dts[i]

            # Use batched time everywhere (your SDE code expects (B,))
            t_b = t.expand(B)

            # Enable gradients wrt state z for divergence
            z = z.detach().requires_grad_(True)

            # 1) Model prediction (needs grad wrt z; DO NOT wrap in no_grad)
            pred_eps = model(z, t_b, labels)

            # 2) eps -> score conversion using marginal std(t)
            # std does not depend on z0 for these SDEs, but marginal_prob signature needs x0.
            _, std_1d = diff_proc.sde.marginal_prob(torch.zeros_like(z), t_b)  # std: (B,)
            std = _broadcast_like(std_1d, z)
            score = -pred_eps / std  # score ≈ ∇ log p_t(z)

            # 3) Probability flow ODE drift: f_ode = f - 0.5 * g^2 * score
            drift, diffusion_1d = diff_proc.sde.sde(z, t_b)  # diffusion: (B,)
            diffusion = _broadcast_like(diffusion_1d, z)
            f_ode = drift - 0.5 * (diffusion ** 2) * score

            # 4) Hutchinson divergence estimate: div(f_ode)(z) = tr(J), J = ∂f_ode/∂z
            # IMPORTANT FIX vs your version:
            #   divergence must be of f_ode as a *function of z* (not a constant tensor).
            div_est = 0.0
            for _ in range(hutchinson_samples):
                eps = torch.randn_like(z)
                vjp = grad((f_ode * eps).sum(), z, create_graph=False, retain_graph=True)[0]
                div_est = div_est + (vjp * eps).sum(dim=tuple(range(1, z.dim())))  # (B,)
            div_est = div_est / float(hutchinson_samples)

            # 5) Euler step forward in time
            with torch.no_grad():
                z = z + f_ode * dt
                logp_correction = logp_correction + div_est * dt

        # Prior term at time T MUST come from the SDE (VE differs!)
        with torch.no_grad():
            zT = z.detach().contiguous()  # makes it contiguous if needed
            logp_T = diff_proc.sde.prior_logp(zT)
            # logp_T = diff_proc.sde.prior_logp(z.detach())  # (B,)
            logp0 = logp_T + logp_correction
            nll = -logp0  # nats

            total_nll += nll.sum().item()
            sumsq_nll += (nll ** 2).sum().item()
            total_count += B

    # Restore model param requires_grad flags
    for p, r in zip(model.parameters(), prev_req):
        p.requires_grad_(r)
    
    mean = total_nll / max(total_count, 1)
    var = (sumsq_nll / total_count) - (mean ** 2)
    std = math.sqrt(var)

    return mean, std, total_count


# --------------- END ODE EXACT COMPUTATION ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=2000, help="Numero di immagini da generare per FID")
    parser.add_argument('--batch-size', type=int, default=32)

    # ----------- ADD PARAMETERS ---------------
    parser.add_argument('--sde-type', type=str, default='subvp')
    # parser.add_argument('--probability-flow', type=bool, default=False)

    parser.add_argument('--probability-flow', action='store_true',
                        help="Enable probability-flow ODE (for ODE NLL / ODE sampling).")
    parser.add_argument('--no-probability-flow', dest='probability_flow', action='store_false')
    parser.set_defaults(probability_flow=False)

    parser.add_argument('--nll-num-images', type=int, default=0,
                    help="Limit number of test images used for ODE NLL. 0 = use all.")
    # ---------- END ADD PARAMETERS ------------

    args = parser.parse_args()

    # Load Config
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override batch size per testing
    cfg['batch_size'] = args.batch_size
    cfg['N'] = cfg.get('n_timesteps', 1000)
    # ----------- ADD PARAMETERS ---------------
    cfg['sde_type'] = args.sde_type
    cfg['probability_flow'] = args.probability_flow
    cfg['nll_num_images'] = args.nll_num_images
    print(f"Running the ODE solver: {cfg['probability_flow']}")
    print(f"The SDE reverse process type is:{cfg['sde_type']}")
    print(f"The SDE reverse process with ODE use this:{cfg['nll_num_images']} number of images.")

    # ---------- END ADD PARAMETERS ------------
    
    def model_fn(x, t, labels):
        # Diffusers expects t to be shape (B,) or a scalar, usually handles it fine.
        with torch.no_grad():
            out = model(x, t, labels)
                
        # Extract the actual tensor from the output object
        if hasattr(out, 'sample'):
            return out.sample
        return out
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=4, out_channels=4).to(device)

    # Setup WandB per il test
    wandb.init(project="LDM_Testing", config=cfg, name=f"FID_IS_Evaluation_{os.path.basename(args.checkpoint_path)}")

    # 1. Setup Data & Model
    # Usiamo setup parziale per avere i loader
    # Nota: Setup restituisce LDM module, ma qui ricostruiamo per controllo fine
    _, _, _, test_loader = setup(cfg, args.data_path, device)

    
    if not os.path.dirname(args.checkpoint_path):
        args.checkpoint_path = os.path.join("checkpoints", "weights", args.checkpoint_path)

    # unet = load_model_from_local(model, args.checkpoint_path, device)
    print("Checkpoints loaded with success!")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    load_model_from_local(model, args.checkpoint_path, device)

    # Setup Processi
    diff_proc = Diffusion_Processes(cfg)
    vae_encoder, vae_decoder = get_vae_encoder_func(device)
    vae_scale_factor = cfg.get('vae_scale_factor', 0.18215)
    latent_size = cfg['image_size'] // cfg['vae_factor']
    print(f"Setting VAE factor: {cfg['vae_factor']}, Latent Size: {latent_size}")
    # -----------------------------------------------------------
    # 2. Calcolo NLL (Test Loss)
    # -----------------------------------------------------------
    # if not cfg['probability_flow']:
    #     nll_score = avg_model_error(unet, test_loader, diff_proc, device, vae_encoder=vae_encoder, vae_scale_factor=vae_scale_factor)
    #     print(f"Test NLL (Loss): {nll_score:.4f}")
    #     wandb.log({"test/nll_loss": nll_score})

    # -----------------------------------------------------------
    # 2.1 Exact NLL computation
    # -----------------------------------------------------------
    if cfg['probability_flow']:
        test_loader_nll = test_loader
        if cfg['nll_num_images']:
            ds = test_loader.dataset
            n = min(cfg['nll_num_images'], len(ds))
            ds_nll = Subset(ds, list(range(n)))
            test_loader_nll = DataLoader(ds_nll,
                                        batch_size = cfg['batch_size'],
                                        shuffle = False,
                                        num_workers=0,
                                        pin_memory=getattr(test_loader, "pin_memory", False),
                                        collate_fn=test_loader.collate_fn,
                                        drop_last=False)


        mean_nll, std_nll, n = calculate_nll_ode(
        model=model,
        loader=test_loader_nll,
        diff_proc=diff_proc,
        device=device,
        vae_encoder=vae_encoder,
        vae_scale_factor=vae_scale_factor,
        n_steps=cfg.get("n_timesteps", 1000),
        eps_time=1e-5,
        )
        print(f"Test NLL (ODE, nats): {mean_nll:.4f} ± {std_nll:.4f} (N={n})")
        wandb.log({"test/nll_ode_mean_nats": mean_nll, "test/nll_ode_std_nats": std_nll, "test/nll_ode_N": n})

    # -----------------------------------------------------------
    # 3. Setup Metriche (FID e IS)
    # -----------------------------------------------------------
    # FID richiede uint8 [0, 255]
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)

    print(">>> Accumulando statistiche immagini REALI per FID...")
    # Limitiamo il numero di immagini reali per FID per velocità (es. stesso num di generated)
    real_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Real Images"):
            if isinstance(batch, list): img = batch[0]
            else: img = batch
            
            img = img.to(device)
            # Mappa [-1, 1] -> [0, 255] byte
            img_uint8 = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            
            fid_metric.update(img_uint8, real=True)
            
            real_count += img.shape[0]
            if real_count >= args.num_samples:
                break

    # -----------------------------------------------------------
    # 4. Generazione Immagini e Logging Denoising
    # -----------------------------------------------------------
    print(f">>> Generando {args.num_samples} immagini FAKE per FID/IS...")
    
    generated_count = 0
    num_batches_gen = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    # Definisci la shape
    # Attenzione: latent_channels, image_size dipendono se stai generando nello spazio latente o pixel
    # Se UNet lavora sui pixel (come sembra da CelebA dataset loading):
    shape = (args.batch_size, cfg['latent_channels'], latent_size, latent_size)

    for i in tqdm(range(num_batches_gen), desc="Generation"):
        # Solo per il primo batch attiviamo la visualizzazione step-by-step
        cb_fn = log_denoising_step_wandb if i == 0 else None
        
        # Generazione attributi casuali (se condizionale)
        labels = None
        if cfg.get('conditional', False):
            # Esempio: sample random labels o zero labels
            # Qui usiamo zero per semplicità o random binary
            labels = torch.randint(0, 2, (args.batch_size, cfg['num_attributes'])).float().to(device)

        # Reverse Process
        print("Initiate the reverse process")
        samples = diff_proc.reverse_process(
            model=model_fn,
            shape=shape,
            device=device,
            labels=labels,
            callback_fn=cb_fn
            # probability_flow=cfg['probability_flow']
        )
        with torch.no_grad():
            samples = vae_decoder(samples / vae_scale_factor)
        
        
        # Post-process per metriche
        print(f"Image minimum before clamping and multipling: {samples.min()}")
        samples_uint8 = ((samples + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        
        # Update Metriche
        fid_metric.update(samples_uint8, real=False)
        is_metric.update(samples_uint8)
        
        generated_count += shape[0]
        
        # Log del batch finale generato
        if i == 0:
            grid = torchvision.utils.make_grid(samples_uint8[:16], nrow=4)
            wandb.log({"generated_samples/final_batch": wandb.Image(grid)})

    # -----------------------------------------------------------
    # 5. Calcolo Finale e Log
    # -----------------------------------------------------------
    print(">>> Calcolando FID e IS finali...")
    fid_score = fid_metric.compute().item()
    is_score, is_std = is_metric.compute()
    
    print(f"FID: {fid_score:.4f}")
    print(f"IS: {is_score.item():.4f} +/- {is_std.item():.4f}")
    
    wandb.log({
        "test/fid": fid_score,
        "test/inception_score": is_score.item(),
        "test/inception_std": is_std.item()
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()

# python test_revised.py --config-path="experiments/base_config.yaml" --checkpoint-path="checkpoints/weights/last-v6.ckpt" --data-path="./data" --probability-flow --sde-type="ve" --num-samples=8 --batch-size=8 --nll-num-images=8
