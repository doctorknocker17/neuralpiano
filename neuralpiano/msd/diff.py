import pytorch_lightning as pl
from torch.optim import AdamW
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
from models.diff_decoder import Midi2SpecDiffusionConfig, Midi2SpecDiffusionModel
from .mel_db import MelToDB
from diffusers import DDPMScheduler


class DiffusionLM(pl.LightningModule):
    """
    A diffusion model for converting MIDI to spectrograms with context conditioning.
    
    This model implements a diffusion process to generate spectrograms from MIDI inputs,
    supporting classifier-free guidance and optional context conditioning.
    """
    
    def __init__(
        self,
        num_emb: int = 900,
        output_dim: int = 128,
        max_input_length: int = 2048,
        max_output_length: int = 512,
        emb_dim: int = 512,
        dim_feedforward: int = 1024,
        nhead: int = 6,
        num_layers: int = 8,
        cfg_dropout: float = 0.1,
        cfg_weighting: float = 2.0,
        with_context: bool = False,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        num_train_timesteps: int = 1000,
        return_dict: bool = False,
        gradient_clip_val: float = 0.5,
        **mel_kwargs
    ) -> None:
        """
        Initialize the DiffusionLM model.
        
        Args:
            num_emb: Number of MIDI token embeddings
            output_dim: Dimension of output spectrogram (typically n_mels)
            max_input_length: Maximum MIDI sequence length
            max_output_length: Maximum spectrogram sequence length
            emb_dim: Embedding dimension
            dim_feedforward: Dimension of feedforward network
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            cfg_dropout: Dropout probability for classifier-free guidance
            cfg_weighting: Weighting factor for classifier-free guidance
            with_context: Whether to use context conditioning
            dropout: General dropout probability
            layer_norm_eps: Epsilon for layer normalization
            norm_first: Whether to apply layer norm before attention
            num_train_timesteps: Number of diffusion timesteps
            return_dict: Whether to return dictionaries from model
            gradient_clip_val: Value for gradient clipping
            **mel_kwargs: Additional arguments for MelToDB
        """
        super().__init__()
        self.save_hyperparameters(ignore=['return_dict', 'layer_norm_eps', 'norm_first'])
        
        # Store dB range for spectrogram clamping
        self.min_db = -11.512925
        self.max_db = 2.5241776
        
        self.cfg_dropout = cfg_dropout
        self.cfg_weighting = cfg_weighting
        self.output_dim = output_dim
        self.gradient_clip_val = gradient_clip_val
        
        # Build model configuration
        config = Midi2SpecDiffusionConfig(
            num_embeddings=num_emb,
            output_dim=output_dim,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            emb_dim=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            with_context=with_context,
            return_dict=return_dict,
        )
        
        # Instantiate the model
        self.model = Midi2SpecDiffusionModel(config)
        
        # Initialize Mel spectrogram converter
        self.mel = MelToDB(
            sample_rate=mel_kwargs.get('sample_rate', 44100),
            n_fft=mel_kwargs.get('n_fft', 2048),
            hop_length=mel_kwargs.get('hop_length', 512),
            win_length=mel_kwargs.get('win_length', 2048),
            n_mels=mel_kwargs.get('n_mels', 128),
            f_max=mel_kwargs.get('f_max', 22050.0),
            f_min=mel_kwargs.get('f_min', 0.0),
            normalize=mel_kwargs.get('normalize', True),
        )
        
        # Initialize diffusion scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
            clip_sample=False
        )
        
        # Store eps for potential use
        self.mel_eps = mel_kwargs.get('eps', 1e-10)

    def _check_finite(self, tensor: Tensor, name: str, batch_idx: int = None) -> Tensor:
        """Handle NaN/Inf values and apply appropriate clamping"""
        # Always fix NaN/Inf values
        if not torch.isfinite(tensor).all():
            self.log(f"warning/{name}_nan", 1.0, prog_bar=False, logger=True, batch_size=1)
            tensor = torch.nan_to_num(
                tensor, 
                nan=0.0, 
                posinf=self.max_db if "spec" in name or "z_t" in name else 5.0,
                neginf=self.min_db if "spec" in name or "z_t" in name else -5.0
            )
        
        # ONLY clamp actual spectrogram values (NOT noise predictions)
        if any(x in name for x in ['spec', 'z_t']):
            if (tensor < self.min_db).any() or (tensor > self.max_db).any():
                self.log(f"warning/{name}_out_of_range", 1.0, prog_bar=False, logger=True, batch_size=1)
            tensor = torch.clamp(tensor, min=self.min_db, max=self.max_db)
        
        # For noise predictions, allow wider range but prevent extreme outliers
        elif any(x in name for x in ['noise', 'pred']):
            # Standard Gaussian should be within ~[-5,5] (99.99994% of values)
            tensor = torch.clamp(tensor, min=-5.0, max=5.0)
        
        return tensor

    def _prepare_context(self, wav_context: Tensor = None, mel_context: Tensor = None) -> Tensor:
        """
        Prepare context tensor from either waveform or mel spectrogram.
        
        Args:
            wav_context: Optional waveform context [B, T_ctx]
            mel_context: Optional mel spectrogram context [B, T_ctx, n_mels]
            
        Returns:
            Prepared context tensor [B, T_ctx, n_mels] or None
        """
        if wav_context is not None:
            spec_ctx = self.mel(wav_context)  # [B, n_mels, T_ctx]
            return spec_ctx.transpose(1, 2)  # [B, T_ctx, n_mels]
        return mel_context

    def get_training_inputs(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generate noisy versions of input for training.
        
        Args:
            x: Clean input spectrograms [B, T, n_mels]
            
        Returns:
            Tuple of (noisy_input, timesteps, noise)
        """
        batch_size = x.size(0)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=x.device,
            dtype=torch.long
        )
        noise = torch.randn_like(x)
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        return x_noisy, timesteps, noise

    def forward(
        self,
        midi: Tensor,
        seq_length: int = 256,
        mel_context: Tensor = None,
        wav_context: Tensor = None,
        rescale: bool = True,
        T: int = 1000,
        verbose: bool = True
    ) -> Tensor:
        """
        Generate spectrogram from MIDI input.
        
        Args:
            midi: MIDI token sequence [B, seq_len]
            seq_length: Length of output spectrogram
            mel_context: Optional mel spectrogram context [B, T_ctx, n_mels]
            wav_context: Optional waveform context [B, T_ctx]
            rescale: Whether to convert from dB to amplitude
            T: Number of diffusion steps
            verbose: Whether to show progress bar
            
        Returns:
            Generated spectrogram [B, seq_length, n_mels]
        """
        # Prepare context
        context = self._prepare_context(wav_context, mel_context)
        
        batch_size = midi.size(0)
        # Start from random noise
        z_t = torch.randn(batch_size, seq_length, self.output_dim, device=self.device)
        
        # Classifier-free guidance setup
        midi = midi.repeat(2, 1)
        if context is not None:
            context = context.repeat(2, 1, 1)
        
        # Create dropout mask for classifier-free guidance
        dropout_mask = torch.zeros(2 * batch_size, device=self.device, dtype=torch.bool)
        dropout_mask[batch_size:] = True
        
        # Set up diffusion timesteps
        self.scheduler.set_timesteps(T)
        
        # Reverse diffusion process
        for step in tqdm(range(T-1, -1, -1), disable=not verbose):
            t_val = self.scheduler.timesteps[step]
            t_tensor = torch.full((2 * batch_size,), t_val, device=self.device, dtype=torch.long)
            
            # Get noise prediction
            noise_pred = self.model(
                midi,
                z_t.repeat(2, 1, 1),
                t_tensor,
                context,
                dropout_mask=dropout_mask
            )
            
            # Handle numerical instability
            noise_pred = self._check_finite(noise_pred, "noise_pred")
            
            # Apply classifier-free guidance
            cond, uncond = noise_pred.chunk(2, dim=0)
            # Consider adding a scheduler for this value
            
            if self.current_epoch < 5:
                effective_cfg = 1.0 + (self.cfg_weighting - 1.0) * (self.current_epoch / 5)
            else:
                effective_cfg = self.cfg_weighting
                
            guided = cond * effective_cfg + uncond * (1 - effective_cfg)
            
            z_t = self.scheduler.step(guided, t_val, z_t).prev_sample
            z_t = torch.clamp(z_t, min=self.min_db, max=self.max_db)  # Add this line
            
            # Additional numerical stability
            z_t = self._check_finite(z_t, f"z_t_step_{step}")
        
        # Convert from dB to amplitude if requested
        return self.mel.reverse(z_t) if rescale else z_t

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Perform a single training step.
        
        Args:
            batch: Input batch containing (midi, wav, [context_wav])
            batch_idx: Index of current batch
            
        Returns:
            Computed loss
        """
        
        midi, wav, *rest = batch
        wav = self._check_finite(wav, "wav", batch_idx)
        spec = self._check_finite(self.mel(wav).transpose(1, 2), "spec", batch_idx)
        
        context = None
        if rest:
            context_wav = self._check_finite(rest[0], "context_wav", batch_idx)
            context = self._check_finite(self.mel(context_wav).transpose(1, 2), "context_spec", batch_idx)
        
        batch_size = midi.size(0)
        dropout_mask = torch.bernoulli(torch.full((batch_size,), 1.0 - self.cfg_dropout, 
                                                 device=self.device)).bool()
        
        # Get timesteps and store them
        z_t, timesteps, noise = self.get_training_inputs(spec)
        self._current_timesteps = timesteps  # Store for gradient monitoring
        
        # Compute per-example loss (this DOES have batch dimension)
        noise_hat = self.model(midi, z_t, timesteps, context, dropout_mask=dropout_mask)
        noise_hat = self._check_finite(noise_hat, "noise_hat", batch_idx)
        noise = self._check_finite(noise, "noise", batch_idx)
        
        # Calculate loss per example in batch
        per_example_loss = F.l1_loss(noise_hat, noise, reduction='none').mean(dim=[1, 2])
        self._current_per_example_loss = per_example_loss
        
        # Calculate overall loss
        loss = per_example_loss.mean()
        self.log('loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        
        # Log basic timestep statistics
        self.log("timestep/mean", timesteps.float().mean(), sync_dist=True, on_step=True)
        self.log("timestep/std", timesteps.float().std(), sync_dist=True, on_step=True)

        # Log value ranges for debugging
        self.log("stats/z_t_min", z_t.min(), sync_dist=True, on_step=True, on_epoch=True)
        self.log("stats/z_t_max", z_t.max(), sync_dist=True, on_step=True, on_epoch=True)
        self.log("stats/z_t_range", (spec.max() - spec.min()), sync_dist=True, on_step=True)
        self.log("stats/noise_range", (noise.max() - noise.min()), sync_dist=True, on_step=True)
        self.log("stats/noise_hat_min", noise_hat.min(), sync_dist=True, on_step=True, on_epoch=True)
        self.log("stats/noise_hat_max", noise_hat.max(), sync_dist=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5, weight_decay=1e-2)
        
        # Create a single scheduler that handles both warmup and custom adjustment
        def lr_lambda(current_step):
            # Warmup phase (0-2000 steps)
            if current_step < 2000:
                return 0.001 + (current_step / 2000) * 0.999
            
            # After warmup, apply timestep-based adjustment
            progress = min((current_step - 2000) / 8000, 1.0)
            timestep_factor = 1.0 - 0.5 * progress
            return timestep_factor
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm"
        }

    def on_before_optimizer_step(self, optimizer, metrics_dict=None):
        """Apply timestep-conditioned gradient clipping"""
        if not hasattr(self, '_current_timesteps') or self._current_timesteps.numel() == 0:
            self.clip_gradients(optimizer, gradient_clip_val=float(self.hparams.gradient_clip_val), 
                               gradient_clip_algorithm="norm")
            return
            
        timesteps = self._current_timesteps
        
        # Calculate timestep-specific clip value
        # Early timesteps (high noise) need more aggressive clipping
        timestep_factor = (timesteps.float().mean() / self.scheduler.config.num_train_timesteps).item()
        clip_val = self.hparams.gradient_clip_val * (0.5 + 0.5 * timestep_factor)
        
        # Ensure clip_val is a Python float
        clip_val = float(clip_val)
        
        self.clip_gradients(optimizer, gradient_clip_val=clip_val, 
                           gradient_clip_algorithm="norm")
        
        # Log the actual clip value used
        self.log("training/clip_val", clip_val, sync_dist=True, on_step=True)

    def on_after_backward(self):
        # Calculate total gradient norm
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, sync_dist=True, on_step=True)
        
        # Log gradient statistics
        max_grad = 0
        min_grad = float('inf')
        all_grads = []
        for p in self.parameters():
            if p.grad is not None:
                grad_abs = p.grad.data.abs()
                max_grad = max(max_grad, grad_abs.max().item())
                if (grad_abs > 0).any():
                    min_grad = min(min_grad, grad_abs[grad_abs > 0].min().item())
                all_grads.append(grad_abs.flatten())
        
        if all_grads:
            all_grads = torch.cat(all_grads)
            
            # If tensor is too large, sample a subset for statistics
            MAX_GRADIENTS_FOR_STATS = 1_000_000  # Keep this manageable
            if all_grads.numel() > MAX_GRADIENTS_FOR_STATS:
                # Randomly sample indices without replacement
                indices = torch.randperm(all_grads.numel(), device=all_grads.device)[:MAX_GRADIENTS_FOR_STATS]
                sampled_grads = all_grads[indices]
            else:
                sampled_grads = all_grads
            
            # Log the basic statistics we can measure
            self.log("grad/max_value", max_grad, sync_dist=True, on_step=True)
            self.log("grad/min_nonzero", min_grad, sync_dist=True, on_step=True)
            self.log("grad/median", torch.median(sampled_grads).item(), sync_dist=True, on_step=True)
            self.log("grad/95th_percentile", torch.quantile(sampled_grads, 0.95).item(), sync_dist=True, on_step=True)
        
        # If we have timestep information, log correlations with loss
        if hasattr(self, '_current_timesteps') and hasattr(self, '_current_per_example_loss'):
            timesteps = self._current_timesteps.float()
            losses = self._current_per_example_loss
            
            # Calculate correlation between timestep and loss
            timestep_mean = timesteps.mean()
            loss_mean = losses.mean()
            cov = ((timesteps - timestep_mean) * (losses - loss_mean)).mean()
            timestep_std = timesteps.std(unbiased=False)
            loss_std = losses.std(unbiased=False)
            
            if timestep_std > 0 and loss_std > 0:
                corr = cov / (timestep_std * loss_std)
                self.log("timestep/loss_corr", corr.item(), sync_dist=True, on_step=True)
            
            # Log loss at different timestep quantiles
            sorted_indices = torch.argsort(timesteps)
            timesteps_sorted = timesteps[sorted_indices]
            losses_sorted = losses[sorted_indices]
            
            for q in [0.25, 0.5, 0.75]:
                idx = int(len(timesteps) * q)
                self.log(f"loss/timestep_q{q*100:.0f}", losses_sorted[idx].item(), sync_dist=True, on_step=True)