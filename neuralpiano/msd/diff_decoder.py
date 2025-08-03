import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

# -----------------------------------------------------------------------------
# 1. CONFIG (with backward compatibility for parameter names)
# -----------------------------------------------------------------------------
class Midi2SpecDiffusionConfig(PretrainedConfig, ConfigMixin):
    model_type = "midi2spec_diffusion"
    config_name = "Midi2SpecDiffusionConfig"
    
    @register_to_config
    def __init__(
        self,
        num_embeddings:      int   = 128,
        num_emb:             Optional[int] = None,  # Backward compatibility
        output_dim:          int   = 80,
        max_input_length:    int   = 512,
        max_output_length:   int   = 1024,
        emb_dim:             int   = 256,
        nhead:               int   = 8,
        num_encoder_layers:  int   = 6,
        num_decoder_layers:  int   = 6,
        num_layers:          Optional[int] = None,  # Backward compatibility
        dropout:           float   = 0.1,
        with_context:      bool    = False,
        return_dict:       bool    = False,
        **kwargs,
    ):
        # Handle backward compatibility for parameter names
        if num_emb is not None and num_embeddings == 128:
            num_embeddings = num_emb
        if num_layers is not None:
            num_encoder_layers = num_layers
            num_decoder_layers = num_layers
            
        super().__init__(**kwargs)
        self.num_embeddings     = num_embeddings
        self.output_dim         = output_dim
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length
        self.emb_dim            = emb_dim
        self.nhead              = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout            = dropout
        self.with_context       = with_context
        self.return_dict        = return_dict

# -----------------------------------------------------------------------------
# 2. UTILS (with fixed FiLM layer)
# -----------------------------------------------------------------------------
def geglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class FiLM(nn.Module):
    """Fixed to properly handle broadcasting across sequence length"""
    def __init__(self, cond_dim: int, emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, emb_dim * 2)
        # Proper initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Fixed dimension handling:
        - x: [batch_size, seq_len, emb_dim]
        - cond: [batch_size, cond_dim] or [batch_size, 1, cond_dim]
        """
        # Ensure cond has the right shape for broadcasting
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # [batch_size, 1, cond_dim]
        
        # Project and split
        params = self.linear(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * 0.1
        # Now gamma and beta have shape [batch_size, 1, emb_dim]
        # which will broadcast correctly across seq_len dimension
        return x * (1 + gamma) + beta

class SinusoidalPosEmb(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        pos      = torch.arange(max_len).unsqueeze(1)
        emb      = torch.zeros(max_len, dim)
        emb[:, 0::2] = torch.sin(pos * inv_freq)
        emb[:, 1::2] = torch.cos(pos * inv_freq)
        self.register_buffer("emb", emb, persistent=False)
    
    def forward(self, length: int) -> torch.Tensor:
        return self.emb[:length].unsqueeze(0)

class CustomTimeEmbed(nn.Module):
    """
    Map timesteps → sinusoidal → MLP → cond vector (emb_dim*4).
    Fixed to handle various input shapes correctly.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        cond_dim = emb_dim * 4
        self.lin1 = nn.Linear(emb_dim,    cond_dim)
        self.lin2 = nn.Linear(cond_dim,   cond_dim)
        self.half = emb_dim // 2
        self.norm = nn.LayerNorm(cond_dim)  # Add to __init__
        
        # Proper initialization
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        
    
    def forward(self, timesteps: torch.LongTensor) -> torch.Tensor:
        # Handle various input shapes (scalar, 1D, 2D)
        timesteps = torch.clamp(timesteps, 0, 999)
        original_shape = timesteps.shape
        timesteps = timesteps.reshape(-1)  # Flatten to 1D
        
        freqs = torch.exp(-math.log(10000) / (self.half - 1) *
                          torch.arange(self.half, device=timesteps.device))
        args  = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        sin   = torch.sin(args)
        cos   = torch.cos(args)
        x     = torch.cat([sin, cos], dim=1)
        x     = F.silu(self.lin1(x))
        result = self.lin2(x)

        # Then in forward():
        result = self.norm(result)
        result = result * 0.01  # Reduce from 0.1 to 0.01
        
        return result.view(*original_shape, -1)

# -----------------------------------------------------------------------------
# 3. DECODER LAYER WITH FiLM (fixed dimension handling)
# -----------------------------------------------------------------------------
class DiffDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, emb_dim: int, nhead: int, cond_dim: int, num_decoder_layers: int, dropout: float = 0.1):
        super().__init__(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.film1 = FiLM(cond_dim, emb_dim)
        self.film2 = FiLM(cond_dim, emb_dim)
        # Store the total number of decoder layers for residual scaling
        self.num_decoder_layers = num_decoder_layers
        # Precompute scaling factor (1/sqrt(n) is standard for transformers)
        self.residual_scale = 1.0 / math.sqrt(num_decoder_layers)

        self.gamma1 = nn.Parameter(torch.ones((emb_dim,)) * 0.1)  # Small positive value
        self.gamma2 = nn.Parameter(torch.ones((emb_dim,)) * 0.1)  # Small positive value

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cond: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ) -> torch.Tensor:
        # Self-attn with FiLM
        x = tgt
        sa_input = self.film1(self.norm1(x), cond)
        sa_input = torch.clamp(sa_input, min=-10.0, max=10.0)
        sa_output = self._sa_block(sa_input, tgt_mask, tgt_key_padding_mask)
        x = x + sa_output * self.gamma1
        x = x + sa_output * self.residual_scale
        
        # Encoder-decoder attention
        mha_output = self._mha_block(self.norm2(x), memory,
                                memory_mask, memory_key_padding_mask)
        x = x + mha_output * self.gamma2
        x = x + mha_output * self.residual_scale
        
        # Feed-forward with FiLM
        ff_input = self.film2(self.norm3(x), cond)
        ff_output = self._ff_block(ff_input)
        return x + ff_output * self.residual_scale
        
# -----------------------------------------------------------------------------
# 4. TRANSFORMER WITH FIXED MASK HANDLING
# -----------------------------------------------------------------------------
class DiffTransformer(nn.Module):
    def __init__(self, cfg: Midi2SpecDiffusionConfig):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.emb_dim * 4,
            dropout=cfg.dropout,
            activation=F.gelu,  # Use function, not string
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.num_encoder_layers,
            norm=nn.LayerNorm(cfg.emb_dim),
        )
        
        if cfg.with_context:
            self.context_encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=cfg.num_encoder_layers,
                norm=nn.LayerNorm(cfg.emb_dim),
            )
        
        # Create decoder layers with the total number of layers
        self.layers = nn.ModuleList([
            DiffDecoderLayer(
                emb_dim=cfg.emb_dim,
                nhead=cfg.nhead,
                cond_dim=cfg.emb_dim * 4,
                num_decoder_layers=cfg.num_decoder_layers,  # Pass the total count
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_decoder_layers)
        ])
        
        self.norm = nn.LayerNorm(cfg.emb_dim)
        
        # Proper initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    @staticmethod
    def _sanitize(kpm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if kpm is None:
            return None
        if kpm.dtype != torch.bool:
            kpm = kpm.bool()
        if kpm.ndim == 3:
            kpm = kpm.squeeze(1).all(dim=-1)
        return kpm
    
    @staticmethod
    def _match_len(src: torch.Tensor, kpm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if kpm is None:
            return None
        B, T = src.shape[:2]
        cur  = kpm.shape[1]
        if cur < T:
            pad = torch.zeros(B, T - cur, dtype=torch.bool, device=kpm.device)
            return torch.cat([kpm, pad], dim=1)
        return kpm[:, :T]
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        cond: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        dropout_mask: Optional[torch.BoolTensor] = None,
        src_mask=None,
        tgt_mask=None,
        ctx_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        ctx_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ) -> torch.Tensor:
        # 1) pad-mask sanitation
        src_kpm = self._match_len(src, self._sanitize(src_key_padding_mask))
        ctx_kpm = self._match_len(ctx, self._sanitize(ctx_key_padding_mask)) if ctx is not None else None
        tgt_kpm = self._match_len(tgt, self._sanitize(tgt_key_padding_mask))
        
        # 2) encode inputs with optional dropout
        if dropout_mask is not None:
            memory = torch.zeros_like(src)
            ctx_mem = torch.zeros_like(ctx) if ctx is not None else None
            sel = ~dropout_mask
            
            # CRITICAL FIX: Check if sel has any True values before processing
            if sel.any():
                memory[sel] = self.encoder(
                    src[sel], mask=src_mask, src_key_padding_mask=src_kpm[sel] if src_kpm is not None else None
                )
                if ctx is not None and hasattr(self, "context_encoder"):
                    ctx_mem[sel] = self.context_encoder(
                        ctx[sel], mask=ctx_mask, src_key_padding_mask=ctx_kpm[sel] if ctx_kpm is not None else None
                    )
            # If sel is all False, memory remains zeros (which is fine)
        else:
            memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_kpm)
            ctx_mem = self.context_encoder(ctx, mask=ctx_mask, src_key_padding_mask=ctx_kpm) \
                if hasattr(self, "context_encoder") and ctx is not None else None
        
        # 3) fuse context
        if ctx is not None and ctx_mem is not None:
            memory = torch.cat([memory, ctx_mem], dim=1)
            if src_kpm is not None and ctx_kpm is not None:
                src_kpm = torch.cat([src_kpm, ctx_kpm], dim=1)
        
        # 4) decode
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                cond,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=src_kpm,
            )
        return self.norm(x)

# -----------------------------------------------------------------------------
# 5. HF WRAPPER + DDP FIX (with fixed time embedding handling)
# -----------------------------------------------------------------------------
class MJSDiffusionOutput(ModelOutput):
    predicted_specs: torch.FloatTensor = None

class Midi2SpecDiffusionModel(PreTrainedModel):
    config_class    = Midi2SpecDiffusionConfig
    _timestep_input = "timesteps"
    
    def __init__(self, config: Midi2SpecDiffusionConfig):
        super().__init__(config)
        # embeddings
        self.token_emb = nn.Embedding(config.num_embeddings, config.emb_dim)
        self.pos_in    = SinusoidalPosEmb(config.max_input_length,  config.emb_dim)
        self.pos_out   = SinusoidalPosEmb(config.max_output_length, config.emb_dim)
        
        # time→conditioning - FIXED: Better shape handling
        self.time_embed = CustomTimeEmbed(config.emb_dim)
        
        # transformer core
        self.transformer = DiffTransformer(config)
        
        # spec projection
        self.spec_in   = nn.Linear(config.output_dim, config.emb_dim)
        self.spec_out = nn.Sequential(
            nn.Linear(config.emb_dim, config.output_dim),
            nn.Tanh()  # CRITICAL: constrain output to [-1,1] range
        )
        
        self.return_dict = config.return_dict
        self.post_init()
    
    def forward(
        self,
        midi_tokens:    torch.LongTensor,
        specs:          torch.FloatTensor,
        timesteps:      torch.LongTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        spec_mask:      Optional[torch.BoolTensor] = None,
        context:        Optional[torch.FloatTensor] = None,
        context_mask:   Optional[torch.BoolTensor] = None,
        dropout_mask:   Optional[torch.BoolTensor] = None,
        src_mask:       Optional[torch.Tensor]      = None,
        tgt_mask:       Optional[torch.Tensor]      = None,
        ctx_mask:       Optional[torch.Tensor]      = None,
        memory_mask:    Optional[torch.Tensor]      = None,
        memory_key_padding_mask: Optional[torch.BoolTensor] = None,
        return_dict:    Optional[bool]              = None,
    ) -> torch.FloatTensor or Dict[str, Any]:
        return_dict = return_dict if return_dict is not None else self.return_dict
        
        # encode MIDI + positional
        x_m = self.token_emb(midi_tokens)
        x_m = x_m + self.pos_in(x_m.size(1))
        
        # encode specs + positional
        x_s = self.spec_in(specs)
        x_s = x_s + self.pos_out(x_s.size(1))
        
        # FIXED: Proper timestep handling to ensure correct dimensions
        # This is the critical fix for your error
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)  # Make it 1D if scalar
        
        # Ensure timesteps is 1D for proper processing
        timesteps = timesteps.reshape(-1).to(dtype=torch.long, device=x_m.device)
        
        # Get time embeddings - now properly shaped
        cond = self.time_embed(timesteps)
        
        # If needed, add sequence dimension for broadcasting
        # This ensures cond has shape [batch_size, 1, emb_dim*4]
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        
        # optional context
        x_ctx = None
        if context is not None:
            x_ctx = self.spec_in(context) + self.pos_out(context.size(1))
        
        # core transformer
        hidden = self.transformer(
            src=                    x_m,
            tgt=                    x_s,
            cond=                   cond,
            ctx=                    x_ctx,
            dropout_mask=           dropout_mask,
            src_mask=               src_mask,
            tgt_mask=               tgt_mask,
            ctx_mask=               ctx_mask,
            memory_mask=            memory_mask,
            src_key_padding_mask=   attention_mask,
            ctx_key_padding_mask=   context_mask,
            tgt_key_padding_mask=   spec_mask,
            memory_key_padding_mask= memory_key_padding_mask,
        )
        
        predicted_specs = self.spec_out(hidden)
        
        if not return_dict:
            return predicted_specs
            
        return MJSDiffusionOutput(predicted_specs=predicted_specs)
    
    def configure_ddp(self, ddp_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Enable finding unused parameters under DDP to avoid runtime errors."""
        ddp_kwargs["find_unused_parameters"] = True
        return ddp_kwargs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[str] = None, *model_args, **kwargs):
        """
        Override from_pretrained to handle configuration parameter mismatches
        """
        # Check if config is provided directly
        config = kwargs.pop("config", None)
        
        # If config is a path, load it
        if isinstance(config, str):
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(config, **kwargs)
        
        # If no config provided, try to load from model path
        if config is None and pretrained_model_name_or_path is not None:
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except Exception as e:
                raise ValueError(f"Could not load configuration: {str(e)}") from e
        
        # Handle parameter name mismatches
        if config:
            # Create a copy of config to avoid modifying the original
            config_dict = config.to_dict()
            
            # Handle deprecated parameter names
            if "num_emb" in config_dict and "num_embeddings" not in config_dict:
                config_dict["num_embeddings"] = config_dict.pop("num_emb")
            
            if "num_layers" in config_dict:
                config_dict["num_encoder_layers"] = config_dict.get("num_encoder_layers", config_dict["num_layers"])
                config_dict["num_decoder_layers"] = config_dict.get("num_decoder_layers", config_dict["num_layers"])
            
            # Create new config with corrected parameters
            config = Midi2SpecDiffusionConfig.from_dict(config_dict)
        
        # Set config in kwargs for the parent method
        kwargs["config"] = config
        
        # Call the parent method
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)