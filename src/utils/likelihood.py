import torch
from subVP_SDE import subVP_SDE
from Configurations import ForwardConfig

# if we want to integrate it properly we should add reduce in cfg and number of grid steps
class Likelihood:
    def __init__(self, sde: subVP_sde, cfg: ForwardConfig):
        self.sde = sde
        self.cfg = cfg
    
    def alpha_squared(self, t: torch.Tensor):
        """
        Compute the square of the alpha original, which is the mean coefficient of the subVP SDE process
        """
        return sde.mean_coeff(t) ** 2
    
    def time_proposal(self, num_grid: int = 10000, device = 'cpu'):
        """
        We are returning the grid, the discretized probability and the cumulative probability.
        """
        grid = torch.linspace(self.cfg.eps, T-self.cfg.eps, num_grid, device = device)
        w = self.sde.get_g_squared(grid) / self.sde.alpha_squared(grid) + 1e-20
        pdf = w / (w.sum() + 1e-20) # normalizing the weights sampled 
        cdf = torch.cumsum(pdf, dim=0) # cumulative sum of probabilities (pdf)
        return grid, cdf, pdf # we are assuming Z=1, because we have normalized, so the constant that ensures \sum pdf = 1 has already been considered. Otherwise we would have set Z = 1/w.sum()

    def sample_t_from_proposal(self, grid: torch.Tensor, dim:int, gen = None):
        u = torch.rand(dim, device = grid.device, generator = gen)# we sample randomly
        idx = torch.searchsorted(cdf, u, right = True).clamp(max = grid.numel()-1) # we search for the closest value in the discreate version
        return grid[idx]
    
    def lw_loss_with_is(self, model: nn.Module, x: torch.Tensor, grid, cdf, reduce = 'mean'):
        """
        Computing the loss with likelihood weighting and adjusting the loos also for importance sampling.
        """
        device = next(model.parameteres()).device
        
        shape = x.shape[0]
        t = self.sample_t_from_proposal(grid, shape, cdf)
        z = torch.randn_like(x)
        
        mean = self.sde.mean_coeff(t)[:, None, None, None] * x
        std = torch.sqrt(self.sde.var(t))[:, None, None, None]

        x_t = mean + std * z 
        score = model(x_t, t)

        resid = score + z / (std + 1e-12)
        per_dimensions = resid.reshape(shape, -1)**2
        
        w_is = 1.0 * self.alpha_squared(t, beta0, beta1) # T = 1.0
        per_sample = per_dim.mean(dim=1) * w_is
        
        return per_sample.mean() if reduce == 'mean' else per_sample.sum()


"""
Example of implementation of the LW and IS in the loss computation.

grid, cdf, pdf = make_time_proposal(beta0, beta1, n_grid=20000, device=device)

model = UNetScoreModel(...).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Loop over epochs and batches
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = batch.to(device)

        # Compute the IS+LW DSM loss
        loss = lw_dsm_loss_with_is(
            model,
            batch,
            beta0, beta1,
            grid=grid,
            cdf=cdf,
            T=1.0,                     # diffusion end time
            reduce='mean'
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.6f}")
"""