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
        if schedule not in ['linear', 'exponential']:
            raise ValueError("The inserted scheduler is wrong.")
        self.schedule = schedule


class LikelihoodConfig:
    """
    Parameters configuartion for the likelihood process
    """
    def __init__(
        self,
        output_path="likelihoods.pt",
        scores = "scores.pt",
        t0: float = 1.0,
        t1: float = 0.0,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        N: int = 1000,
        schedule: str = "linear",
        device = None,
        dtype = None,
    ):
        self.output_path = output_path
        self.scores = scores
        self.t0 = t0
        self.t1 = t1
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.steps = N
        if schedule not in ['linear','exponential']:
            raise ValueError("The inserted scheduler is wrong.")
        
        self.schedule = schedule
        self.device = device
        self.dtype = dtype