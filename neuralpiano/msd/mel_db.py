import torch
from torch import nn
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from typing import Optional, Tuple, Dict, Any

# Global cache dictionaries
MEL_BASIS_CACHE: Dict[Tuple, torch.Tensor] = {}
HANN_WINDOW_CACHE: Dict[Tuple, torch.Tensor] = {}

def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Apply log compression to magnitude spectrogram with safe clamping."""
    return torch.log(torch.clamp(magnitudes, min=1e-5))

def spectral_de_normalize_torch(log_magnitudes: torch.Tensor) -> torch.Tensor:
    """Convert log-magnitude spectrogram back to linear scale."""
    return torch.exp(log_magnitudes)


class MelToDB(nn.Module):
    """
    PyTorch module for computing log-magnitude mel-spectrograms with optional normalization.
    
    This implementation closely follows BigVGAN's mel_spectrogram function, using:
    - Manual reflect padding
    - torch.stft for STFT computation
    - librosa's mel filterbank
    
    Attributes:
        sample_rate: Audio sampling rate in Hz
        n_fft: FFT window size
        hop_length: Hop length between successive frames
        win_length: Window length (defaults to n_fft if None)
        n_mels: Number of mel filterbanks
        f_min: Minimum frequency for mel filters
        f_max: Maximum frequency for mel filters (None for Nyquist)
        min_level_db: Minimum dB value for normalization
        max_level_db: Maximum dB value for normalization
        normalize: Whether to scale output to [-1, 1] range
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        min_level_db: float = -11.512925,  # log(1e-5)
        max_level_db: float = 2.5241776,   # log(12.5)
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize
        self.min_db = min_level_db
        self.max_db = max_level_db
        self.eps = 1e-10
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters for correctness."""
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.win_length <= 0:
            raise ValueError(f"win_length must be positive, got {self.win_length}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if self.f_min < 0:
            raise ValueError(f"f_min must be non-negative, got {self.f_min}")
        if self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError(f"f_max ({self.f_max}) must be greater than f_min ({self.f_min})")
    
    def _get_cache_key(self, device: torch.device) -> Tuple:
        """Generate a cache key based on parameters and device."""
        return (
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            str(device)
        )
    
    def _get_or_create_filters(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create mel basis and Hann window for the current configuration."""
        cache_key = self._get_cache_key(device)
        
        # Get or create mel basis
        if cache_key not in MEL_BASIS_CACHE:
            # Generate mel filterbank using librosa
            mel_basis = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max
            )
            # Convert to torch tensor and move to device
            MEL_BASIS_CACHE[cache_key] = torch.from_numpy(mel_basis).float().to(device)
            HANN_WINDOW_CACHE[cache_key] = torch.hann_window(self.win_length, device=device)
        
        return MEL_BASIS_CACHE[cache_key], HANN_WINDOW_CACHE[cache_key]
    
    def _handle_nonfinite(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Handle NaN/Inf values in a tensor with appropriate warnings."""
        if not torch.isfinite(tensor).all():
            nonfinite_mask = ~torch.isfinite(tensor)
            num_nonfinite = nonfinite_mask.sum().item()
            total_elements = tensor.numel()
            
            print(f"Warning: {num_nonfinite}/{total_elements} non-finite values found in {name}. "
                  "Replacing with safe values.")
            
            # Replace NaN with 0, Inf with max/min values
            tensor = torch.nan_to_num(
                tensor, 
                nan=0.0, 
                posinf=torch.finfo(tensor.dtype).max * 0.9,
                neginf=torch.finfo(tensor.dtype).min * 0.9
            )
        return tensor
    
    def _normalize_to_range(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        """Normalize log-mel spectrogram to [-1, 1] range."""
        # Scale from [min_db, max_db] to [0, 1], then to [-1, 1]
        denom = (self.max_db - self.min_db) + self.eps
        normalized = (log_mel_spec - self.min_db) / denom * 2.0 - 1.0
        return torch.clamp(normalized, min=-1.0, max=1.0)
    
    def _denormalize_from_range(self, normalized_spec: torch.Tensor) -> torch.Tensor:
        """Convert normalized spectrogram back to log-magnitude scale."""
        # Reverse the [-1, 1] scaling to get back to log scale
        return (normalized_spec + 1.0) / 2.0 * (self.max_db - self.min_db) + self.min_db
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-magnitude mel-spectrogram from audio waveform.
        
        Args:
            x: Input audio tensor of shape [B, T] or [B, 1, T]
        
        Returns:
            Log-magnitude mel-spectrogram of shape [B, n_mels, T_frames]
            If normalize=True, values are scaled to [-1, 1]
            If normalize=False, values are in log scale (log(magnitude))
        """
        # Validate input
        if x.dim() not in [2, 3]:
            raise ValueError(f"Input must be 2D [B, T] or 3D [B, 1, T], got {x.dim()}D tensor")
        
        # Handle channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] → [B, 1, T]
        
        if x.shape[1] != 1:
            raise ValueError(f"Input must have 1 channel dimension, got shape {x.shape}")
        
        x = x.squeeze(1)  # [B, 1, T] → [B, T]
        device = x.device
        
        # Check for non-finite values
        x = self._handle_nonfinite(x, "input audio")
        
        # Get or create filters
        mel_basis, hann_window = self._get_or_create_filters(device)
        
        # Calculate padding
        padding_amount = (self.n_fft - self.hop_length) // 2
        x_padded = torch.nn.functional.pad(
            x.unsqueeze(1),  # Add channel dim for padding
            (padding_amount, padding_amount),
            mode="reflect"
        ).squeeze(1)  # [B, T_padded]
        
        # STFT
        complex_spec = torch.stft(
            x_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Compute magnitude spectrogram
        magnitude_spec = torch.sqrt(torch.view_as_real(complex_spec).pow(2).sum(-1) + 1e-9)
        
        # Apply mel filterbank
        mel_spec = torch.matmul(mel_basis, magnitude_spec)
        
        # Apply log compression
        log_mel_spec = spectral_normalize_torch(mel_spec)
        log_mel_spec = self._handle_nonfinite(log_mel_spec, "log mel spectrogram")
        
        # Apply normalization if requested
        if self.normalize:
            return self._normalize_to_range(log_mel_spec)
        
        return log_mel_spec
    
    def reverse(self, normalized_or_log: torch.Tensor) -> torch.Tensor:
        """
        Reverse the transformation to get log-magnitude spectrogram.
        
        Args:
            normalized_or_log: Normalized spectrogram (if normalize=True) 
                               or log-magnitude spectrogram (if normalize=False)
        
        Returns:
            Log-magnitude spectrogram in original scale
        """
        if self.normalize:
            # Input is normalized to [-1, 1], convert back to log scale
            normalized_or_log = torch.clamp(normalized_or_log, min=-1.0, max=1.0)
            return self._denormalize_from_range(normalized_or_log)
        return normalized_or_log
    
    def reverse_to_linear(self, normalized_or_log: torch.Tensor) -> torch.Tensor:
        """
        Convert to linear magnitude spectrogram.
        
        Args:
            normalized_or_log: Normalized spectrogram or log-magnitude spectrogram
            
        Returns:
            Linear magnitude spectrogram
        """
        log_mel = self.reverse(normalized_or_log)
        return spectral_de_normalize_torch(log_mel)