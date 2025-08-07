import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from diff import DiffusionLM
from data import ConcatData

torch.set_float32_matmul_precision('high')

def cli_main():
    cli = LightningCLI(
        trainer_defaults={
            'accelerator': 'gpu',
            'strategy': 'ddp',
            'log_every_n_steps': 1,
            'callbacks': [
                ModelCheckpoint(
                    save_top_k=-1,
                    save_last=True,
                    every_n_train_steps=1879,
                    filename='{epoch}-{step}-{loss}',
                ),
            ]
        }
    )


if __name__ == "__main__":
    cli_main()
