import pytorch_lightning as pl

import torch
from torch import Tensor, nn

import torch.nn.functional as F

import math
from tqdm import tqdm

from diff_decoder import MIDI2SpecDiff

from mel import MelSpectrogram

from scaler import get_scaler

#=============================================================

def snr2as(snr: Tensor):
    snr_p1 = snr + 1
    return torch.sqrt(snr / snr_p1), snr_p1.reciprocal()


def log_snr2as(log_snr: Tensor):
    var = (-log_snr).sigmoid()
    return (1 - var).sqrt(), var


def log_snr2logas(log_snr: Tensor):
    log_var = -F.softplus(log_snr)
    return 0.5 * (log_snr + log_var), log_var


class DiffusionLM(pl.LightningModule):
    def __init__(self,
                 num_emb: int = 687,
                 output_dim: int = 128,
                 max_input_length: int = 896,
                 max_output_length: int = 257,
                 emb_dim: int = 768,
                 dim_feedforward: int = 1536,
                 nhead: int = 8,
                 head_dim: int = 64,
                 num_layers: int = 8,
                 cfg_dropout: float = 0.1,
                 cfg_weighting: float = 2.0,
                 with_context: bool = False,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = True,
                 **mel_kwargs) -> None:
        super().__init__()

        self.cfg_dropout = cfg_dropout
        self.cfg_weighting = cfg_weighting
        self.output_dim = output_dim

        self.model = MIDI2SpecDiff(
            num_emb=num_emb, output_dim=output_dim, max_input_length=max_input_length,
            max_output_length=max_output_length, emb_dim=emb_dim, nhead=nhead, with_context=with_context,
            head_dim=head_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout,
            layer_norm_eps=layer_norm_eps, norm_first=norm_first, dim_feedforward=dim_feedforward
        )

        # Mels and scaler..
        self.mel = nn.Sequential(
            MelSpectrogram(),
            get_scaler()
        )
    
    def get_log_snr(self, t):

        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp t to avoid boundaries
        t = t.clamp(1e-4, 1.0 - 1e-4)
    
        # Precompute constants
        logsnr_max = 20.0
        logsnr_min = -20.0
        b = math.atan(math.exp(-0.5 * logsnr_max))  # ~4.54e-5
        a = math.atan(math.exp(-0.5 * logsnr_min)) - b  # ~π/2 - 4.54e-5
    
        angle = a * t + b
    
        # 🔑 Critical: clamp angle to (0, π/2)
        min_angle = 1e-4
        max_angle = math.pi / 2 - 1e-4
        angle = angle.clamp(min_angle, max_angle)
    
        tan_val = torch.tan(angle)
    
        # Avoid overflow in log
        tan_val = tan_val.clamp(1e-8, 1e8)  # reasonable dynamic range
    
        log_snr = -2.0 * torch.log(tan_val)
    
        # Final clamp to avoid extreme values that could destabilize training
        log_snr = log_snr.clamp(-20.0, 20.0)
    
        return log_snr

    def forward(self, midi: Tensor, seq_length=256, mel_context=None, wav_context=None, rescale=True, T=1000, verbose=True):
        if wav_context is not None:
            context = self.mel(wav_context)
            
        elif mel_context is not None:
            context = mel_context
            
        else:
            context = None

        t = torch.linspace(0, 1, T).to(self.device)
        log_snr = self.get_log_snr(t)
        log_alpha, log_var = log_snr2logas(log_snr)

        var = log_var.exp()
        alpha = log_alpha.exp()
        alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
        c = -torch.expm1(log_snr[1:] - log_snr[:-1])
        c.relu_()

        z_t = torch.randn(
            midi.shape[0], seq_length, self.output_dim, device=self.device)
        
        z_t = torch.nan_to_num(z_t, nan=0.0, posinf=0.0, neginf=0.0)
        z_t = z_t.clamp(-5.0, 5.0)
        
        dropout_mask = torch.tensor(
            [0] * midi.shape[0] + [1] * midi.shape[0]).bool().to(self.device)
        t = torch.broadcast_to(t, (midi.shape[0] * 2, T))
        midi = midi.repeat(2, 1)
        
        if context is not None:
            context = context.repeat(2, 1, 1)

        for t_idx in tqdm(range(T - 1, -1, -1), disable=not verbose):
            s_idx = t_idx - 1
            noise_hat = self.model(midi, z_t.repeat(
                2, 1, 1), t[:, t_idx], context, dropout_mask=dropout_mask)
           
            noise_hat = torch.nan_to_num(noise_hat, nan=0.0, posinf=0.0, neginf=0.0)
            
            cond_noise_hat, uncond_noise_hat = noise_hat.chunk(2, dim=0)
            noise_hat = cond_noise_hat * self.cfg_weighting + \
                uncond_noise_hat * (1 - self.cfg_weighting)
            
            noise_hat = torch.nan_to_num(noise_hat, nan=0.0, posinf=0.0, neginf=0.0)

            noise_hat = noise_hat.clamp_(
                (z_t - alpha[t_idx]) * var[t_idx].rsqrt(),
                (alpha[t_idx] + z_t) * var[t_idx].rsqrt(),
            )

            if s_idx >= 0:
                mu = (z_t - var[t_idx].sqrt() * c[s_idx]
                      * noise_hat) * alpha_st[s_idx]
                z_t = mu + (var[s_idx] * c[s_idx]).sqrt() * \
                    torch.randn_like(z_t)
                continue
          
            final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]

        if rescale:
            final = self.mel[1].reverse(final)

        final = torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
            
        return final

    def get_training_inputs(self, x: torch.Tensor, uniform: bool = False):
        
        N = x.shape[0]
        
        if uniform:
            t = torch.linspace(0, 1, N).to(x.device)
            
        else:
            t = x.new_empty(N).uniform_(0, 1)
            
        log_snr = self.get_log_snr(t)
        alpha, var = log_snr2as(log_snr)
        sigma = var.sqrt()
        noise = torch.randn_like(x)
        z_t = x * alpha[:, None, None] + sigma[:, None, None] * noise
        
        return z_t, t, noise
    
    def training_step(self, batch, batch_idx):
        midi, wav, *_ = batch
        
        spec = self.mel(wav)

        if len(_) > 0:
            context = _[0]
            context = self.mel(context)

        else:
            context = None
            
        N = midi.shape[0]
        
        dropout_mask = spec.new_empty(N).bernoulli_(self.cfg_dropout).bool()
        z_t, t, noise = self.get_training_inputs(spec)
        
        z_t = torch.nan_to_num(z_t, nan=0.0, posinf=0.0, neginf=0.0)
        z_t = z_t.clamp(-5.0, 5.0)
        
        noise_hat = self.model(midi, z_t, t, context,
                               dropout_mask=dropout_mask)
        
        noise = torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
        noise_hat = torch.nan_to_num(noise_hat, nan=0.0, posinf=0.0, neginf=0.0)
        
        loss = F.l1_loss(noise_hat, noise)

        self.log('loss', loss, prog_bar=True, sync_dist=True)

        if batch_idx % 10 == 0:
            self.log('mel/min', spec.min(), sync_dist=True)
            self.log('mel/max', spec.max(), sync_dist=True)
            self.log('z_t/std', z_t.std(), sync_dist=True)
            self.log('noise_hat/std', noise_hat.std(), sync_dist=True)
                
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch == 1 and not getattr(self, '_scaler_frozen', False):
            self.mel[1].frozen = True
            self._scaler_frozen = True
            print(f"✅ Scaler frozen at epoch 1: min={self.mel[1].min.item():.3f}, max={self.mel[1].max.item():.3f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=1000
        )
        return [optimizer], [scheduler]