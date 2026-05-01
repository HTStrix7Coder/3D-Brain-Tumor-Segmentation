import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, Convolution, Upsample
from monai.networks.nets.swin_unetr import SwinTransformerBlock
from monai.networks.layers.factories import Conv, Norm, Pool

class ModalityGating(nn.Module):
    """
    Adaptive Modality Gating (Squeeze-and-Excitation style).
    Learns to weight the 4 MRI modalities (T1n, T1c, T2w, T2f) 
    based on the content of the current patch.
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2 if in_channels > 2 else in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2 if in_channels > 2 else in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, return_weights=False):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        weights = self.fc(y)
        y = weights.view(b, c, 1, 1, 1)
        out = x * y.expand_as(x)
        if return_weights:
            return out, weights
        return out

class VAEBranch(nn.Module):
    """
    VAE Branch for reconstruction regularization.
    Forces the encoder to learn features sufficient to reconstruct the input.
    """
    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        self.recon_decoder = nn.Sequential(
            Convolution(3, in_channels, in_channels, strides=1, kernel_size=3, padding=1),
            Upsample(spatial_dims=3, in_channels=in_channels), 
            Convolution(3, in_channels, in_channels//2, strides=1, kernel_size=3, padding=1),
            Upsample(spatial_dims=3, in_channels=in_channels//2), 
            Convolution(3, in_channels//2, in_channels//4, strides=1, kernel_size=3, padding=1),
            Upsample(spatial_dims=3, in_channels=in_channels//4), 
            Convolution(3, in_channels//4, out_channels, strides=1, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.recon_decoder(x)

class AdvancedBraTSNet(nn.Module):
    """
    Hybrid CNN-Transformer 3D U-Net for BraTS.
    - Modality Gating (Input)
    - Residual CNN Encoder (Levels 0-2)
    - Swin Transformer (Level 3 - Bottleneck)
    - VAE Reconstruction Branch
    """
    def __init__(self, in_channels=4, out_channels=3, features=(32, 64, 128, 256), vae_reg=True):
        super().__init__()
        self.vae_reg = vae_reg
        self.gating = ModalityGating(in_channels)
        
        # Encoder (CNN Levels 0-2)
        self.enc1 = ResidualUnit(3, in_channels, features[0], strides=1, kernel_size=3, padding=1, subunits=2)
        self.down1 = Convolution(3, features[0], features[1], strides=2, kernel_size=3, padding=1)
        
        self.enc2 = ResidualUnit(3, features[1], features[1], strides=1, kernel_size=3, padding=1, subunits=2)
        self.down2 = Convolution(3, features[1], features[2], strides=2, kernel_size=3, padding=1)
        
        self.enc3 = ResidualUnit(3, features[2], features[2], strides=1, kernel_size=3, padding=1, subunits=2)
        self.down3 = Convolution(3, features[2], features[3], strides=2, kernel_size=3, padding=1)
        
        # Bottleneck: Global Self-Attention
        # At 12x12x12, we can afford full global attention (no windows needed)
        self.bottleneck_attention = nn.Sequential(
            Convolution(3, features[3], features[3], strides=1, kernel_size=3, padding=1),
            ResidualUnit(3, features[3], features[3], strides=1, subunits=2),
            # Add a simple Self-Attention mechanism here
        )
        self.attn = nn.MultiheadAttention(embed_dim=features[3], num_heads=8, batch_first=True)
        
        # VAE Branch (Regularization)
        if vae_reg:
            self.vae = VAEBranch(features[3], out_channels=in_channels)
            
        # Decoder (CNN with Skip Connections)
        self.up3 = Upsample(spatial_dims=3, in_channels=features[3], out_channels=features[2])
        self.dec3 = ResidualUnit(3, features[2] * 2, features[2], strides=1, kernel_size=3, padding=1, subunits=2)
        
        self.up2 = Upsample(spatial_dims=3, in_channels=features[2], out_channels=features[1])
        self.dec2 = ResidualUnit(3, features[1] * 2, features[1], strides=1, kernel_size=3, padding=1, subunits=2)
        
        self.up1 = Upsample(spatial_dims=3, in_channels=features[1], out_channels=features[0])
        self.dec1 = ResidualUnit(3, features[0] * 2, features[0], strides=1, kernel_size=3, padding=1, subunits=2)
        
        self.final_conv = Convolution(3, features[0], out_channels, strides=1, kernel_size=1, bias=True, act=None, norm=None)

    def forward(self, x, return_gating=False):
        # 1. Modality Gating
        gating_weights = None
        if return_gating:
            x, gating_weights = self.gating(x, return_weights=True)
        else:
            x = self.gating(x)
        
        # 2. Encoder
        e1 = self.enc1(x)
        x = self.down1(e1)
        
        e2 = self.enc2(x)
        x = self.down2(e2)
        
        e3 = self.enc3(x)
        x = self.down3(e3)
        
        # 3. Transformer/Attention Bottleneck
        x = self.bottleneck_attention(x)
        b, c, h, w, d = x.shape
        # Flatten spatial dims for attention: (B, C, H, W, D) -> (B, H*W*D, C)
        x_flat = x.view(b, c, h * w * d).permute(0, 2, 1)
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        # Reshape back to 3D: (B, H*W*D, C) -> (B, C, H, W, D)
        x = x_attn.permute(0, 2, 1).view(b, c, h, w, d)
        
        # 4. VAE Reconstruction
        vae_out = None
        if self.vae_reg and self.training:
            vae_out = self.vae(x)
            
        # 5. Decoder with Skip Connections
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        seg_out = self.final_conv(x)
        
        if self.vae_reg and self.training:
            return seg_out, vae_out
        
        if return_gating:
            return seg_out, gating_weights
            
        return seg_out
