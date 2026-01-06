import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class GMMDataset:
    """
    Generates samples from a 2D Gaussian Mixture Model.
    Targeted for testing diffusion model mode-recovery.
    """
    def __init__(self, means: list, covs: list, weights: list = None):
        """
        Args:
            means: List of 2-element lists/tensors for the means.
            covs: List of 2x2 covariance matrices (or scalars for isotropic).
            weights: Mixing coefficients. Defaults to uniform.
        """
        self.means = [torch.tensor(m, dtype=torch.float32) for m in means]
        self.covs = [torch.tensor(c, dtype=torch.float32) if torch.is_tensor(c) 
                     else torch.eye(2) * c for c in covs]
        
        if weights is None:
            self.weights = torch.ones(len(means)) / len(means)
        else:
            self.weights = torch.tensor(weights)

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample from the mixture: 
        1. Sample component index k ~ Categorical(weights)
        2. Sample x ~ N(mu_k, sigma_k)
        """
        # Choose which Gaussian to sample from for each point in batch
        indices = torch.multinomial(self.weights, batch_size, replacement=True)
        
        samples = []
        for idx in indices:
            m = self.means[idx]
            c = self.covs[idx]
            # Sample using Cholesky for arbitrary covariance
            L = torch.linalg.cholesky(c)
            z = torch.randn(2)
            x = m + L @ z
            samples.append(x)
            
        return torch.stack(samples), indices

class GMMImageDataset:
    def __init__(self, gmm_dataset, img_size=32, sigma_pixel=4, k=5):
        self.gmm = gmm_dataset
        self.img_size = img_size
        self.sigma_pixel = sigma_pixel
        
        # --- DYNAMIC RANGE CALCULATION ---
        # Collect all means: shape (K, 2)
        all_means = torch.stack(self.gmm.means) 
        
        # Estimate sigma from the diagonal of covariances: shape (K, 2)
        # Using sqrt of variance for each dimension
        all_sigmas = torch.stack([torch.sqrt(torch.diag(c)) for c in self.gmm.covs])
        
        # Calculate the absolute min/max across all components and dimensions
        data_min = torch.min(all_means - k * all_sigmas).item()
        data_max = torch.max(all_means + k * all_sigmas).item()
        
        # Ensure the range is symmetric and has some padding
        limit = max(abs(data_min), abs(data_max))
        self.data_range = (-limit, limit)

    def sample_as_images(self, batch_size: int, channels: int = 4) -> torch.Tensor:
        """
        Samples from GMM and converts coordinates to RGB heatmap images (B, 3, H, W).
        """
        # 1. Sample continuous coordinates and the component indices that produced them
        # Modification: Use a version of sample that returns indices to assign colors

        # GMM 2D sampling
        coords, component_indices = self.gmm.sample(batch_size) 
        
        # 2. Map coordinates to pixel indices (same as before)
        min_r, max_r = self.data_range
        normalized = (coords - min_r) / (max_r - min_r)
        pixel_coords = normalized * (self.img_size - 1)
        pixel_coords = pixel_coords.clamp(0, self.img_size - 1)
        
        batch_imgs = []
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.img_size, device=coords.device), 
            torch.arange(self.img_size, device=coords.device), 
            indexing='ij'
        )
        
        for i in range(batch_size):
            # Calculate spatial blob
            dist_sq = (grid_x - pixel_coords[i, 0])**2 + (grid_y - pixel_coords[i, 1])**2
            blob = torch.exp(-dist_sq / (2 * self.sigma_pixel**2))

            # print("Distance matrix")
            # print(dist_sq)
            
            # Initialize a 3-channel image
            img = torch.zeros((channels, self.img_size, self.img_size), device=coords.device)
            
            # Logic: Assign colors based on component index (e.g., Mode 0=Red, Mode 1=Green, Mode 2=Blue)
            color_idx = component_indices[i] % 3 
            img[color_idx, :, :] = blob 
            
            batch_imgs.append(img)
            
        return torch.stack(batch_imgs) # Resulting shape: (B, 3, H, W)