class ForwardConfig:
    def __init__(
        self,
        input_path="latents.pt",
        output_path="latents_noised.pt",
        t=0.7,
        final=False,
        eps=1e-5,
        beta_min=0.1,
        beta_max=20.0,
        N=1000,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.t = t
        self.final = final
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
