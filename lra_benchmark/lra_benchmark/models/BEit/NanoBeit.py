import timm
import torch.nn as nn
from .base_model import BaseModel


class NanoBEiT(BaseModel):
    def __init__(
        self,
        img_size=32,
        in_channels=3,
        num_classes=10,
    ):
        super().__init__()
        # Smaller configuration of BEiT
        self.model = timm.create_model(
            'beit_base_patch16_224',  # BEiT base model
            img_size=img_size,
            patch_size=1,            # Patch size
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=32,             # Smaller embedding dimension
            depth=3,                  # 3 transformer layers
            num_heads=2,              # 2 attention heads
            mlp_ratio=2.0,            # MLP ratio
            drop_rate=0.1,
            pretrained=False
        )
    
    def forward(self, x):
        return self.model(x)
        
    @property
    def config(self):
        return {
            'name': 'NanoBEiT-p1',
            'embed_dim': self.model.embed_dim,
            'depth': len(self.model.blocks),
            'num_heads': self.model.blocks[0].attn.num_heads,
            'num_params': sum(p.numel() for p in self.parameters())
        }
