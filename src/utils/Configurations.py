from typing import Tuple, List, Optional


class ForwardConfig:
    """
    Configurazione per il processo forward (noising).
    """
    def __init__(
        self,
        t: float = 0.7,
        final: bool = True,
        eps: float = 1e-5,
        closed_formula: bool = True,
        seed: int = 42,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        schedule: str = "linear",
        # campi aggiunti dal YAML
        use_importance_sampling: bool = True,
        latent_channels: int = 4,
        image_size: int = 128,
        vae_scale_factor: float = 0.18215,
        validation_split_ratio: float = 0.2,
        features: Optional[List[int]] = None,
        self_attention: bool = True,
        num_workers: int = 0,
        data_path: str = "./data/coco2017/train2017",
        # training-related (potrebbero non essere usati direttamente qui ma utili a livello globale)
        epochs: int = 100,
        learning_rate: float = 0.0003,
        batch_size: int = 64,
        model: str = "LDM"
    ):
        # originali
        self.t = t
        self.final = final
        self.eps = eps
        self.closed_formula = closed_formula
        self.seed = seed
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        # scheduler validation
        if schedule not in ['linear', 'exponential']:
            raise ValueError("The inserted scheduler is wrong. Use 'linear' or 'exponential'.")
        self.schedule = schedule

        self.use_importance_sampling = use_importance_sampling
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.vae_scale_factor = vae_scale_factor
        self.validation_split_ratio = validation_split_ratio
        self.features = features if features is not None else [128, 256, 512]
        self.self_attention = self_attention
        self.num_workers = num_workers
        self.data_path = data_path

        # training/meta (utile se passi la stessa config in pipeline)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model


class LikelihoodConfig:
    """
    Configurazione per il calcolo delle likelihoods / scoring.
    """
    def __init__(
        self,
        t0: float = 1.0,
        t1: float = 0.0,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        schedule: str = "linear",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        # campi aggiunti dal YAML
        use_importance_sampling: bool = True,
        latent_channels: int = 4,
        image_size: int = 128,
        vae_scale_factor: float = 0.18215,
        validation_split_ratio: float = 0.2,
        features: Optional[List[int]] = None,
        self_attention: bool = True,
        num_workers: int = 0,
        data_path: str = "./data/coco2017/train2017",
        # training/meta
        epochs: int = 50,
        learning_rate: float = 0.0001,
        batch_size: int = 64,
        model: str = "LDM"
    ):

        self.t0 = t0
        self.t1 = t1
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.steps = N

        if schedule not in ['linear', 'exponential']:
            raise ValueError("The inserted scheduler is wrong. Use 'linear' or 'exponential'.")
        self.schedule = schedule

        self.device = device
        self.dtype = dtype

        # YAML-derived
        self.use_importance_sampling = use_importance_sampling
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.vae_scale_factor = vae_scale_factor
        self.validation_split_ratio = validation_split_ratio
        self.features = features if features is not None else [128, 256, 512]
        self.self_attention = self_attention
        self.num_workers = num_workers
        self.data_path = data_path

        # training/meta
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model


class ReverseConfig:
    """
    Configurazione per il processo reverse (sampling / denoising).
    """
    def __init__(
        self,
        output_path: str = "reverse.pt",
        scores: str = "scores.pt",
        t0: float = 1.0,
        t1: float = 0.0,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        schedule: str = "linear",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        shape: Optional[Tuple[int, int, int, int]] = None,
        seed: int = 42,
        # campi aggiunti dal YAML
        use_importance_sampling: bool = True,
        latent_channels: int = 4,
        image_size: int = 128,
        vae_scale_factor: float = 0.18215,
        validation_split_ratio: float = 0.2,
        features: Optional[List[int]] = None,
        self_attention: bool = True,
        num_workers: int = 0,
        data_path: str = "./data/coco2017/train2017",
        # training/meta
        epochs: int = 50,
        learning_rate: float = 0.0001,
        batch_size: int = 64,
        model: str = "LDM"
    ):
        self.output_path = output_path
        self.scores = scores
        self.t0 = t0
        self.t1 = t1
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.steps = N

        if schedule not in ['linear', 'exponential']:
            raise ValueError("The inserted scheduler is wrong. Use 'linear' or 'exponential'.")
        self.schedule = schedule

        self.device = device
        self.dtype = dtype
        self.shape = shape
        self.seed = seed

        # YAML-derived
        self.use_importance_sampling = use_importance_sampling
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.vae_scale_factor = vae_scale_factor
        self.validation_split_ratio = validation_split_ratio
        self.features = features if features is not None else [128, 256, 512]
        self.self_attention = self_attention
        self.num_workers = num_workers
        self.data_path = data_path

        # training/meta
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model
