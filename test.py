import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import wandb
import os
from tqdm import tqdm
import torchvision
# Import dei tuoi moduli
from src.utils.sde_utils import Diffusion_Processes, SubVPSDE, VESDE, VPSDE
from src.utils.utils import setup
from src.utils.utils import log_denoising_step_wandb # La funzione creata sopra
from src.models.UNet import UNet
from src.models.components import EMAModel

def load_model_from_checkpoint(cfg, checkpoint_path, device):
    """Carica il modello e lo stato dallo script di training."""
    
    # 1. Inizializza architettura
    unet = UNet(in_channels=cfg['latent_channels'], 
                model_channels=64, # Assumiamo default o da cfg
                dropout=0.0, 
                num_attributes=cfg.get('num_attributes', 40)).to(device)
    
    # Se usavi EMA, carichiamo EMA, altrimenti UNet base
    # Qui carichiamo direttamente i pesi nel modello
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Gestione delle chiavi 'state_dict' di Lightning
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    
    # Rimuovi prefissi 'unet_model.' o 'ema_model.' se presenti
    for k, v in state_dict.items():
        if 'unet_model.' in k:
            new_state_dict[k.replace('unet_model.', '')] = v
        # Se preferisci testare con i pesi EMA (consigliato per FID):
        # elif 'ema_model.' in k:
        #    new_state_dict[k.replace('ema_model.', '')] = v
        else:
             new_state_dict[k] = v
             
    # Caricamento strict=False per evitare crash su chiavi ausiliarie (loss, etc)
    unet.load_state_dict(new_state_dict, strict=False)
    unet.eval()
    
    return unet

def calculate_nll(model, loader, forward_process, device):
    """
    Calcola la NLL approssimata (o Loss media) sul test set.
    Per SDE esatti servirebbe l'ODE solver, qui usiamo la Loss di training come proxy.
    """
    print(">>> Calcolo NLL (Test Loss)...")
    total_loss = 0
    num_batches = 0
    loss_fn = nn.MSELoss() # O la loss specifica usata nel training (weighted)

    with torch.no_grad():
        for batch in tqdm(loader, desc="NLL Computation"):
            if isinstance(batch, list):
                x = batch[0].to(device)
                y = batch[1].to(device)
            else:
                x = batch.to(device)
                y = None

            # Forward process per ottenere x_t
            t = torch.rand(x.shape[0], device=device) * forward_process.sde.T
            z_t, t, eps = forward_process.forward_process(x, t)
            
            # Prediction
            # NOTA: Assicurati che score_fn sia coerente con come calcoli la loss in training
            # Qui assumiamo output = eps prediction (standard DDPM/VP)
            if forward_process.conditional and y is not None:
                pred = model(z_t, t, y.float())
            else:
                null_y = torch.zeros((x.shape[0], 40), device=device)
                pred = model(z_t, t, null_y)
                
            loss = loss_fn(pred, eps) # Semplificato
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=2000, help="Numero di immagini da generare per FID")
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # Load Config
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override batch size per testing
    cfg['batch_size'] = args.batch_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup WandB per il test
    wandb.init(project="LDM_Testing", config=cfg, name="FID_IS_Evaluation")

    # 1. Setup Data & Model
    # Usiamo setup parziale per avere i loader
    # Nota: Setup restituisce LDM module, ma qui ricostruiamo per controllo fine
    _, _, _, test_loader = setup(cfg, args.data_path, device)
    
    unet = load_model_from_checkpoint(cfg, args.checkpoint_path, device)
    
    # Setup Processi
    diff_proc = Diffusion_Processes(cfg)

    # -----------------------------------------------------------
    # 2. Calcolo NLL (Test Loss)
    # -----------------------------------------------------------
    nll_score = calculate_nll(unet, test_loader, diff_proc, device)
    print(f"Test NLL (Loss): {nll_score:.4f}")
    wandb.log({"test/nll_loss": nll_score})

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
    shape = (args.batch_size, cfg['latent_channels'], cfg['image_size'], cfg['image_size'])

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
        samples = diff_proc.reverse_process(
            model=unet,
            shape=shape,
            device=device,
            labels=labels,
            callback_fn=cb_fn,    # Passiamo la callback
            callback_every_n=50   # Log ogni 50 step
        )
        
        # Post-process per metriche
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

#python test.py --config-path experiments/base_config.yaml --checkpoint-path checkpoints/weights/ --data-path /path/to/celeba