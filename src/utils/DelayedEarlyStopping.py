from pytorch_lightning.callbacks import EarlyStopping

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        # Attiva EarlyStopping solo se superata la start_epoch
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)