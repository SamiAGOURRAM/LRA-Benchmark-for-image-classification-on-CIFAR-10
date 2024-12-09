import timm
import torch.nn as nn
from .base_model import BaseModel


class PicoBEiT(BaseModel):
    def __init__(
        self,
        img_size=32,
        in_channels=3,
        num_classes=10,
    ):
        super().__init__()
        # Smallest possible configuration of BEiT
        self.model = timm.create_model(
            'beit_base_patch16_224',  # BEiT base model
            img_size=img_size,
            patch_size=1,            # Patch size
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=24,             # Tiny embedding dimension
            depth=2,                  # Only 2 transformer layers
            num_heads=2,              # 2 attention heads
            mlp_ratio=1.0,            # 1:1 MLP ratio
            drop_rate=0.1,
            pretrained=False
        )
    
    def forward(self, x):
        return self.model(x)
        
    @property
    def config(self):
        return {
            'name': 'PicoBEiT-p1',
            'embed_dim': self.model.embed_dim,
            'depth': len(self.model.blocks),
            'num_heads': self.model.blocks[0].attn.num_heads,
            'num_params': sum(p.numel() for p in self.parameters())
        }
