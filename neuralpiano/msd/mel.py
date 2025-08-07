import torch
import torch.nn as nn
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

class MelSpectrogram(nn.Module):
    def __init__(self):
        """
        Initialize the MelSpectrogram module with BigVGAN parameters.
        These values are hardcoded from the BigVGAN configuration.
        """
        super(MelSpectrogram, self).__init__()
        # BigVGAN parameters hardcoded from the config
        self.n_fft = 2048
        self.num_mels = 128
        self.sampling_rate = 44100
        self.hop_size = 512
        self.win_size = 2048
        self.fmin = 0
        self.fmax = 22050
        self.center = False
        
        # Cache for mel basis and hann window
        self.mel_basis_cache = {}
        self.hann_window_cache = {}
        
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Generate mel spectrogram from a waveform.
        
        Args:
            wav (torch.Tensor): Input waveform tensor.
            
        Returns:
            torch.Tensor: Mel spectrogram.
        """
        if torch.min(wav) < -1.0:
            print(f"[WARNING] Min value of input waveform signal is {torch.min(wav)}")
        if torch.max(wav) > 1.0:
            print(f"[WARNING] Max value of input waveform signal is {torch.max(wav)}")
            
        device = wav.device
        key = f"{self.n_fft}_{self.num_mels}_{self.sampling_rate}_{self.hop_size}_{self.win_size}_{self.fmin}_{self.fmax}_{device}"
        
        # Initialize cache if needed
        if key not in self.mel_basis_cache:
            mel = librosa_mel_fn(
                sr=self.sampling_rate, 
                n_fft=self.n_fft, 
                n_mels=self.num_mels, 
                fmin=self.fmin, 
                fmax=self.fmax
            )
            self.mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
            self.hann_window_cache[key] = torch.hann_window(self.win_size).to(device)
            
        mel_basis = self.mel_basis_cache[key]
        hann_window = self.hann_window_cache[key]
        
        # Padding
        padding = (self.n_fft - self.hop_size) // 2
        wav = torch.nn.functional.pad(
            wav.unsqueeze(1), (padding, padding), mode="reflect"
        ).squeeze(1)
        
        # STFT
        spec = torch.stft(
            wav,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        # Convert to magnitude
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        
        # Convert to mel
        mel_spec = torch.matmul(mel_basis, spec)
        
        # Normalize
        mel_spec = spectral_normalize_torch(mel_spec)

        mel_spec = mel_spec.clamp(-12.0, 3.0)
        
        return mel_spec.transpose(-1, -2)