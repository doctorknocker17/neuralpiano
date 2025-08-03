import torch
from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

torch.set_float32_matmul_precision('high')

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Create proper callback instances
        default_callbacks = [
            ModelCheckpoint(
                save_top_k=-1,
                save_last=True,
                every_n_train_steps=1000,
                filename='{epoch}-{step}-{loss}',
            ),
            ModelSummary(max_depth=4)
        ]
        
        # Handle user-specified callbacks
        if 'callbacks' in kwargs:
            if kwargs['callbacks'] is None:
                kwargs['callbacks'] = default_callbacks
            else:
                # Ensure callbacks is a list
                if not isinstance(kwargs['callbacks'], list):
                    kwargs['callbacks'] = [kwargs['callbacks']]
                kwargs['callbacks'].extend(default_callbacks)
        else:
            kwargs['callbacks'] = default_callbacks
            
        # Ensure proper DDP configuration
        if 'strategy' not in kwargs or kwargs['strategy'] in [None, 'ddp']:
            kwargs['strategy'] = 'ddp_find_unused_parameters_true'
            
        super().__init__(*args, **kwargs)

def cli_main():
    cli = LightningCLI(
        trainer_class=CustomTrainer,
        trainer_defaults={
            'accelerator': 'gpu',
            'log_every_n_steps': 1,
        }
    )

if __name__ == "__main__":
    cli_main()