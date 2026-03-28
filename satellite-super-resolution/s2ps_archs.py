"""
Multi-Band Architecture Wrappers for BasicSR
==============================================

BasicSR's built-in RRDB (ESRGAN) and SwinIR default to 3 RGB channels.
These wrappers adapt them for multi-band satellite imagery (4-band BGRN,
8-band SuperDove, etc.) via two approaches:

1. **Direct modification** — set ``num_in_ch`` / ``num_out_ch`` in the
   native arch config (simplest, works for RRDB and SwinIR).
2. **Band-adapter wrappers** — add learnable 1×1 conv layers to map
   N bands → 3 channels → backbone → 3 channels → N bands.  Useful
   for loading pretrained 3-channel weights.

YAML usage::

    network_g:
      type: MultiBandRRDBNet   # 4-band ESRGAN
      num_in_ch: 4
      num_out_ch: 4
      num_feat: 64
      num_block: 23
      num_grow_ch: 32
      scale: 3

    network_g:
      type: MultiBandSwinIR    # 4-band SwinIR
      num_in_ch: 4
      num_out_ch: 4
      upscale: 3
      img_size: 64
      window_size: 8
      depths: [6, 6, 6, 6, 6, 6]
      num_heads: [6, 6, 6, 6, 6, 6]
      embed_dim: 180
      mlp_ratio: 2
      upsampler: pixelshuffle
      resi_connection: 1conv

    network_g:
      type: BandAdapterNet     # any 3-ch backbone with band adapters
      num_bands: 4
      pretrained_3ch: true     # load pretrained RGB weights
      backbone:
        type: RRDBNet
        num_in_ch: 3
        num_out_ch: 3
        ...
"""

import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


# ====================================================================== #
#  Direct multi-band RRDB (ESRGAN generator)                              #
# ====================================================================== #

@ARCH_REGISTRY.register()
class MultiBandRRDBNet(nn.Module):
    """
    RRDB-Net that natively accepts N input/output channels.

    Identical to ``basicsr.archs.rrdbnet_arch.RRDBNet`` but with
    explicit ``num_in_ch`` / ``num_out_ch`` defaults set to 4 for
    satellite multispectral imagery.
    """

    def __init__(
        self,
        num_in_ch: int = 4,
        num_out_ch: int = 4,
        scale: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        self.net = RRDBNet(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ====================================================================== #
#  Direct multi-band SwinIR                                               #
# ====================================================================== #

@ARCH_REGISTRY.register()
class MultiBandSwinIR(nn.Module):
    """
    SwinIR wrapper that passes through ``num_in_ch`` / ``num_out_ch``
    to support multi-spectral input.

    All SwinIR hyperparameters are forwarded; only ``num_in_ch`` and
    ``num_out_ch`` default to 4 instead of 3.
    """

    def __init__(
        self,
        num_in_ch: int = 4,
        num_out_ch: int = 4,
        upscale: int = 3,
        img_size: int = 64,
        window_size: int = 8,
        depths: list = None,
        num_heads: list = None,
        embed_dim: int = 180,
        mlp_ratio: float = 2.0,
        upsampler: str = 'pixelshuffle',
        resi_connection: str = '1conv',
        **kwargs,
    ):
        super().__init__()

        if depths is None:
            depths = [6, 6, 6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6, 6, 6]

        from basicsr.archs.swinir_arch import SwinIR
        self.net = SwinIR(
            upscale=upscale,
            in_chans=num_in_ch,
            img_size=img_size,
            window_size=window_size,
            img_range=1.0,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection=resi_connection,
            **kwargs,
        )
        # SwinIR output channels = in_chans by default for SR
        # Patch it if num_out_ch != num_in_ch
        if num_out_ch != num_in_ch:
            # Replace the final conv
            if hasattr(self.net, 'conv_last'):
                in_f = self.net.conv_last.in_channels
                self.net.conv_last = nn.Conv2d(
                    in_f, num_out_ch, 3, 1, 1
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ====================================================================== #
#  Band Adapter — use pretrained 3-ch weights with N-band data           #
# ====================================================================== #

@ARCH_REGISTRY.register()
class BandAdapterNet(nn.Module):
    """
    Adds learnable 1×1 conv adapters around a 3-channel backbone.

    Architecture::

        Input (B, N, H, W)
          → 1×1 Conv  N → 3          [adapter_in]
          → Backbone   3 → 3 (×scale) [pretrained]
          → 1×1 Conv  3 → N          [adapter_out]
        Output (B, N, H*scale, W*scale)

    This lets you load pretrained ESRGAN/SwinIR/HAT weights trained on
    RGB data and fine-tune on multispectral.  During training, you can
    freeze the backbone initially and train only adapters, then unfreeze
    everything for full fine-tuning.

    YAML::

        network_g:
          type: BandAdapterNet
          num_bands: 4
          freeze_backbone_epochs: 5  # optional
          backbone:
            type: RRDBNet
            num_in_ch: 3
            num_out_ch: 3
            scale: 3
            num_feat: 64
            num_block: 23
            num_grow_ch: 32
    """

    def __init__(
        self,
        num_bands: int = 4,
        backbone: dict = None,
        freeze_backbone_epochs: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.freeze_backbone_epochs = freeze_backbone_epochs

        # Input adapter: N bands → 3 channels
        self.adapter_in = nn.Sequential(
            nn.Conv2d(num_bands, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output adapter: 3 channels → N bands
        self.adapter_out = nn.Conv2d(3, num_bands, 1, bias=False)

        # Build backbone from registry
        if backbone is None:
            raise ValueError('BandAdapterNet requires a `backbone` config.')

        from basicsr.utils.registry import ARCH_REGISTRY as AR
        backbone_type = backbone.pop('type')
        backbone_cls = AR.get(backbone_type)
        self.backbone = backbone_cls(**backbone)

        # Initialise adapters close to identity for NIR passthrough
        self._init_adapters()

    def _init_adapters(self):
        """Initialize adapters so RGB channels pass through ~identity."""
        with torch.no_grad():
            # adapter_in: first 3 output channels = identity for RGB
            w_in = self.adapter_in[0].weight  # (3, N, 1, 1)
            nn.init.zeros_(w_in)
            for i in range(min(3, self.num_bands)):
                w_in[i, i, 0, 0] = 1.0

            # adapter_out: first 3 input channels = identity
            w_out = self.adapter_out.weight  # (N, 3, 1, 1)
            nn.init.zeros_(w_out)
            for i in range(min(3, self.num_bands)):
                w_out[i, i, 0, 0] = 1.0

    def set_epoch(self, epoch: int):
        """Call from training loop to manage backbone freezing schedule."""
        if epoch < self.freeze_backbone_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, H, W) → (B, 3, H, W)
        x3 = self.adapter_in(x)
        # (B, 3, H, W) → (B, 3, H*s, W*s)
        out3 = self.backbone(x3)
        # (B, 3, H*s, W*s) → (B, N, H*s, W*s)
        out = self.adapter_out(out3)

        # Global residual with upsampled input
        x_up = nn.functional.interpolate(
            x, scale_factor=out.shape[-1] / x.shape[-1],
            mode='bilinear', align_corners=False,
        )
        return out + x_up


# ====================================================================== #
#  Lightweight EDSR for quick experiments (no BasicSR dependency)         #
# ====================================================================== #

@ARCH_REGISTRY.register()
class MultiBandEDSR(nn.Module):
    """
    Multi-band EDSR — a fast, lightweight SR backbone for debugging
    and baseline comparisons.  ~1-5M params depending on config.

    YAML::

        network_g:
          type: MultiBandEDSR
          num_in_ch: 4
          num_out_ch: 4
          num_feat: 64
          num_block: 16
          scale: 3
    """

    def __init__(
        self,
        num_in_ch: int = 4,
        num_out_ch: int = 4,
        num_feat: int = 64,
        num_block: int = 16,
        scale: int = 3,
    ):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Body — residual blocks
        body = []
        for _ in range(num_block):
            body.append(_ResBlock(num_feat))
        body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body = nn.Sequential(*body)

        # Upsampler (sub-pixel convolution)
        up = []
        if scale in (2, 4, 8):
            for _ in range(int(np.log2(scale)) if scale > 1 else 1):
                up.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
                up.append(nn.PixelShuffle(2))
                up.append(nn.LeakyReLU(0.2, True))
        elif scale == 3:
            up.append(nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1))
            up.append(nn.PixelShuffle(3))
            up.append(nn.LeakyReLU(0.2, True))
        self.upsampler = nn.Sequential(*up)

        # Tail
        self.tail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        body_out = self.body(feat)
        feat = feat + body_out  # global residual
        feat = self.upsampler(feat)
        out = self.tail(feat)

        # Add upsampled input as global skip
        x_up = nn.functional.interpolate(
            x, scale_factor=self.scale,
            mode='bilinear', align_corners=False,
        )
        return out + x_up


class _ResBlock(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


# Needed by MultiBandEDSR
import numpy as np  # noqa: E402
