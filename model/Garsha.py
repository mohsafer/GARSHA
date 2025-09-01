#REFACTOR

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from einops import rearrange
from itertools import chain

from .attn import HA, AttnLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multi-layer attention for time gpatch anomaly detection.
    
    This encoder processes multi-scale patches through multiple attention layers
    to capture both lpatch and gpatch temporal dependencies.
    """
    
    def __init__(
        self, 
        attention_layers: List[nn.Module], 
        norm_layer: Optional[nn.Module] = None,
        apply_final_norm: bool = True
    ):
        """
        Initialize the transformer encoder.
        
        Args:
            attention_layers: List of attention layer modules
            norm_layer: Optional normalization layer
            apply_final_norm: Whether to apply normalization after all layers
        """
        super(TransformerEncoder, self).__init__()
        
        if not attention_layers:
            raise ValueError("At least one attention layer must be provided")
        
        self.attention_layers = nn.ModuleList(attention_layers)
        self.norm = norm_layer
        self.apply_final_norm = apply_final_norm
        self.num_layers = len(attention_layers)

    def forward(
        self, 
        x_patch_size: torch.Tensor, 
        x_patch_num: torch.Tensor, 
        x_ori: torch.Tensor, 
        patch_index: int, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through all attention layers.
        
        Args:
            x_patch_size: Patch-wise input tensor
            x_patch_num: In-patch input tensor  
            x_ori: Original input tensor
            patch_index: Current patch size index
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (gpatch_list, lpatch_list) containing outputs from all layers
        """
        gpatch_outputs = []
        lpatch_outputs = []
        
        for layer_idx, attention_layer in enumerate(self.attention_layers):
            try:
                gpatch, lpatch = attention_layer(
                    x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask
                )
                
                if gpatch is not None:
                    gpatch_outputs.append(gpatch)
                if lpatch is not None:
                    lpatch_outputs.append(lpatch)
                    
            except Exception as e:
                raise RuntimeError(f"Error in attention layer {layer_idx}: {str(e)}")
        
        # Apply final normalization if specified
        if self.apply_final_norm and self.norm is not None:
            # Note: This would require proper tensor handling based on your specific use case
            pass
        
        return gpatch_outputs, lpatch_outputs


class AnomalyDetector(nn.Module):
    
    def __init__(
        self,
        win_size: int,
        enc_in: int,
        c_out: int,
        n_heads: int = 1,
        d_model: int = 256,
        e_layers: int = 3,
        patch_size: List[int] = None,
        channel: int = 55,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = 'gelu',
        output_attention: bool = True,
        use_revin: bool = True
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            win_size: Input window size
            enc_in: Number of input features
            c_out: Number of output features
            n_heads: Number of attention heads
            d_model: Model dimension
            e_layers: Number of encoder layers
            patch_size: List of patch sizes for multi-scale analysis
            channel: Number of channels for attention computation
            d_ff: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function name
            output_attention: Whether to output attention weights
            use_revin: Whether to use RevIN normalization
        """
        super(AnomalyDetector, self).__init__()
        
        # Validate inputs
        self._validate_init_params(win_size, enc_in, c_out, n_heads, d_model, e_layers)
        
        # Set default patch sizes if not provided
        if patch_size is None:
            patch_size = [3, 5, 7]
        
        # Validate patch sizes
        self._validate_patch_sizes(win_size, patch_size)
        
        # Store configuration
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.d_model = d_model
        self.use_revin = use_revin
        
        # Initialize embedding layers for different patch scales
        self._init_patch_embeddings(d_model, dropout)
        
        # Original window embedding
        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
        
        # Initialize the transformer encoder
        self._init_transformer_encoder(
            win_size, patch_size, channel, d_model, n_heads, e_layers, dropout, output_attention
        )
        
        # Output projection layer
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        # Initialize weights
        self._initialize_weights()

    def _validate_init_params(
        self, win_size: int, enc_in: int, c_out: int, n_heads: int, d_model: int, e_layers: int
    ):
        """Validate initialization parameters."""
        if win_size <= 0:
            raise ValueError(f"Window size must be positive, got {win_size}")
        if enc_in <= 0:
            raise ValueError(f"Input features must be positive, got {enc_in}")
        if c_out <= 0:
            raise ValueError(f"Output features must be positive, got {c_out}")
        if n_heads <= 0:
            raise ValueError(f"Number of heads must be positive, got {n_heads}")
        if d_model <= 0:
            raise ValueError(f"Model dimension must be positive, got {d_model}")
        if d_model % n_heads != 0:
            raise ValueError(f"Model dimension ({d_model}) must be divisible by number of heads ({n_heads})")
        if e_layers <= 0:
            raise ValueError(f"Number of encoder layers must be positive, got {e_layers}")

    def _validate_patch_sizes(self, win_size: int, patch_size: List[int]):
        """Validate patch sizes against window size."""
        for i, p_size in enumerate(patch_size):
            if p_size <= 0:
                raise ValueError(f"Patch size {i} must be positive, got {p_size}")
            if win_size % p_size != 0:
                raise ValueError(f"Window size ({win_size}) must be divisible by patch size {i} ({p_size})")

    def _init_patch_embeddings(self, d_model: int, dropout: float):
        """Initialize embedding layers for different patch sizes."""
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        
        for patch_size in self.patch_size:
            # Embedding for patch content (patch_size dimensions)
            self.embedding_patch_size.append(
                DataEmbedding(patch_size, d_model, dropout)
            )
            # Embedding for number of patches (win_size // patch_size dimensions)
            num_patches = self.win_size // patch_size
            self.embedding_patch_num.append(
                DataEmbedding(num_patches, d_model, dropout)
            )

    def _init_transformer_encoder(
        self, 
        win_size: int, 
        patch_size: List[int], 
        channel: int, 
        d_model: int, 
        n_heads: int, 
        e_layers: int, 
        dropout: float, 
        output_attention: bool
    ):
        """Initialize the transformer encoder with attention layers."""
        attention_layers = []
        
        for _ in range(e_layers):
            attention_layer = AttnLayer(
                HA(
                    win_size=win_size,
                    patch_size=patch_size,
                    channel=channel,
                    mask_flag=False,
                    attention_dropout=dropout,
                    output_attention=output_attention
                ),
                d_model=d_model,
                patch_size=patch_size,
                channel=channel,
                n_heads=n_heads,
                win_size=win_size
            )
            attention_layers.append(attention_layer)
        
        self.encoder = TransformerEncoder(
            attention_layers=attention_layers,
            norm_layer=nn.LayerNorm(d_model),
            apply_final_norm=True
        )

    def _initialize_weights(self):
        """Initialize model weights using appropriate strategies."""
        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0.0)

    def _apply_revin_normalization(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """Apply RevIN normalization if enabled."""
        if not self.use_revin:
            return x
        
        _, _, num_features = x.shape
        revin_layer = RevIN(num_features=num_features)
        return revin_layer(x, mode)

    def _process_multi_scale_patches(
        self, x: torch.Tensor, x_ori: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Process input through multiple patch scales.
        
        Args:
            x: Input tensor [B, L, M]
            x_ori: Original embedded input [B, L, d_model]
            
        Returns:
            Tuple of (gpatch_outputs, lpatch_outputs) from all scales
        """
        all_gpatch_outputs = []
        all_lpatch_outputs = []
        
        for patch_index, patch_size in enumerate(self.patch_size):
            try:
                # Create copies for this patch scale
                x_patch_size = x.clone()
                x_patch_num = x.clone()
                
                # Rearrange for patch processing: [B, L, M] -> [B, M, L]
                x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')
                x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')
                
                # Create patches for global patch attention: [B, M, L] -> [B*M, num_patches, patch_size]
                x_patch_size = rearrange(
                    x_patch_size, 'b m (n p) -> (b m) n p', p=patch_size
                )
                x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
                
                # Create patches for local patch attention: [B, M, L] -> [B*M, patch_size, num_patches]  
                x_patch_num = rearrange(
                    x_patch_num, 'b m (p n) -> (b m) p n', p=patch_size
                )
                x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
                
                # Process through encoder
                gpatch_outputs, lpatch_outputs = self.encoder(
                    x_patch_size, x_patch_num, x_ori, patch_index
                )
                
                all_gpatch_outputs.append(gpatch_outputs)
                all_lpatch_outputs.append(lpatch_outputs)
                
            except Exception as e:
                raise RuntimeError(f"Error processing patch scale {patch_index} (size {patch_size}): {str(e)}")
        
        return all_gpatch_outputs, all_lpatch_outputs

    def forward(self, x: torch.Tensor) -> Union[Tuple[List, List], None]:
        """
        Forward pass of the anomaly detector.
        
        Args:
            x: Input tensor of shape [Batch, Length, Features]
            
        Returns:
            Tuple of (gpatch_attention, lpatch_attention) if output_attention=True, else None
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        batch_size, seq_len, num_features = x.shape
        
        if seq_len != self.win_size:
            raise ValueError(f"Expected sequence length {self.win_size}, got {seq_len}")
        
        # Apply RevIN normalization
        x_normalized = self._apply_revin_normalization(x, mode='norm')
        
        # Create original window embedding
        x_ori = self.embedding_window_size(x_normalized)
        
        # Process through multi-scale patches
        gpatch_outputs, lpatch_outputs = self._process_multi_scale_patches(x_normalized, x_ori)
        
        # Flatten nested lists from all patch scales and layers
        flattened_gpatch = list(chain.from_iterable(gpatch_outputs))
        flattened_lpatch = list(chain.from_iterable(lpatch_outputs))
        
        if self.output_attention:
            return flattened_gpatch, flattened_lpatch
        else:
            return None

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[List, List]:
        """
        Get attention maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (gpatch_attention_maps, lpatch_attention_maps)
        """
        original_setting = self.output_attention
        self.output_attention = True
        
        try:
            result = self.forward(x)
            return result if result is not None else ([], [])
        finally:
            self.output_attention = original_setting

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """Get comprehensive model information."""
        return {
            'model_name': 'AnomalyDetector',
            'window_size': self.win_size,
            'patch_sizes': self.patch_size,
            'model_dimension': self.d_model,
            'num_channels': self.channel,
            'total_parameters': self.count_parameters(),
            'encoder_layers': self.encoder.num_layers,
            'output_attention': self.output_attention,
            'use_revin': self.use_revin
        }



def create_anomaly_detector(
    config: dict = None,
    **kwargs
) -> AnomalyDetector:
    """
    Factory function to create an anomaly detector with default or custom configuration.
    
    Args:
        config: Dictionary of configuration parameters
        **kwargs: Additional keyword arguments that override config
        
    Returns:
        Configured AnomalyDetector instance
    """
    default_config = {
        'win_size': 100,
        'enc_in': 7,
        'c_out': 7,
        'n_heads': 8,
        'd_model': 512,
        'e_layers': 3,
        'patch_size': [3, 5, 7],
        'channel': 55,
        'd_ff': 2048,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': True,
        'use_revin': True
    }
    
    if config:
        default_config.update(config)
    
    default_config.update(kwargs)
    
    return AnomalyDetector(**default_config)