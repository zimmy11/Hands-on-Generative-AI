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
