import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LatentDataset(Dataset):
    """
    Datqset of images for training latent diffusion models. 
    Images are preprocessed with transformations
    to ensure consistent size and format.
    Args:
        data_dir (str): Path to the directory containing the immages.
        image_size (int): Output dimension.  
    
    Ritorna:
        torch.Tensor: Trasformed Image (3, H, W) nel range [0, 1].

    """

    def __init__(self, data_dir, image_size=128):
        """
        Init dataset.
        
        Args:
            data_dir (str): Path to the directory containing the immages.
            image_size (int): Output dimension.  
        """
        self.data_dir = data_dir
        
        # We extract all the images paths from the directory
        self.image_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Define transformations
        # Resize, CenterCrop, ToTensor
        # ToTensor: converts PIL Image to torch.Tensor (Range [0, 1])
            
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), 
        ])

    def __len__(self):
        """
        Outputs the number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Transforms and returns the image at index 'idx'.

        Args:
            idx (int): Index of the image to retrieve.
        Outputs:
            torch.Tensor: Transformed Image (3, H, W) in range [0, 1].
        """

        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:

            # To avoid DataLoader crashing due to corrupted images
            print(f"Skipping corrupt file {img_path}: {e}")

            return self.__getitem__((idx + 1) % len(self))
            
        return self.transform(image)