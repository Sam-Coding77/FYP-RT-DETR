import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from .rtdetr_keypoint_head import YOLOXPoseHeadModule
from src.core import register

# Assume YOLOXPoseHeadModule is already defined here (from the earlier step)
@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'keypoint_head']

    def __init__(self, 
                 backbone: nn.Module, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 keypoint_head: YOLOXPoseHeadModule,  # Added keypoint head
                 multi_scale: List[int] = None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.keypoint_head = keypoint_head  # Initialize keypoint head
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        # Backbone and encoder forward pass
        backbone_features = self.backbone(x)
        x = self.encoder(backbone_features)

        # Decoder forward pass
        x = self.decoder(x, targets)
        
        # Keypoint head forward pass (integrating YOLOXPoseHeadModule)
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = self.keypoint_head(backbone_features)

       

        return {
            'decoder_output': x,  # Original RTDETR output
            'cls_scores': cls_scores,  # Classification scores
            'objectnesses': objectnesses,  # Objectness predictions
            'bbox_preds': bbox_preds,  # Bounding box predictions
            'kpt_offsets': kpt_offsets,  # Keypoint offsets
            'kpt_vis': kpt_vis  # Keypoint visibility predictions
        }


    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
