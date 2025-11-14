from typing import Tuple

class ForwardConfig:
    """
    Parameters configuartion for the forward process
    """
    def __init__(
        self,
        input_path="latents.pt",
        output_path="latents_noised.pt",
        t: float = 0.7,
        final: bool = True,
        eps: float = 1e-5,
        closed_formula: bool = True,
        seed: int = 42,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        schedule: str = "linear"
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.t = t
        self.final = final
        self.eps = eps
        self.closed_formula = closed_formula
        self.seed = seed
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
<<<<<<< Updated upstream
        self.schedule = schedule
=======
        if schedule not in ['linear', 'exponential']:
            raise ValueError("The inserted scheduler is wrong.")
        self.schedule = schedule

class ReverseConfig:
    """
    Parameters configuartion for the reverse process
    """
    def __init__(
        self,
        output_path="latents_denoised.pt",
        scores = "scores.pt",
        t0: float = 1.0,
        t1: float = 0.0,
        n_corr: int = 1,
        target_snr: float = 0.16,
        seed: int = 42,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        schedule: str = "linear",
        device = None,
        dtype = None,
        shape: Tuple[int, int, int, int] = Tuple[15, 3, 32, 32]
        rev_type: str = "sde"
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.scores = scores
        self.t0 = t0
        self.t1 = t1
        self.n_corr = n_corr
        self.target_snr = target_snr
        self.seed = seed
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.steps = N
        if schedule not in ['linear','exponential']:
            raise ValueError("The inserted scheduler is wrong.")
        
        self.schedule = schedule
        self.device = device
        self.dtype = dtype
        self.shape = shape
        if rev_type not in ['sde', 'ode']:
            raise ValueError("The inserted reversion type is wrong.")
        self.rev_type = rev_type
>>>>>>> Stashed changes
