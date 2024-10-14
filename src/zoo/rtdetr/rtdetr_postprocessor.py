"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register


__all__ = ['RTDETRPostProcessor']


@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, num_keypoints=17) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}, num_keypoints={self.num_keypoints}'
    
    def forward(self, outputs, orig_target_sizes):
        """Process the model outputs to extract bounding boxes and keypoints with confidence scores."""
        
        # Extract the logits, bounding boxes, and keypoints from the model output
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        keypoints = outputs['pred_keypoints']  # New addition for keypoints
        
        # Rescale the bounding boxes to the original target sizes
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # Rescale the keypoints to the original image sizes
        keypoints_pred = keypoints * orig_target_sizes.unsqueeze(1).unsqueeze(1)  # Assuming the keypoints are normalized

        if self.use_focal_loss:
            # Process scores using sigmoid for focal loss
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            keypoints_pred = keypoints_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.num_keypoints, 2))

        else:
            # Process scores using softmax
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
                keypoints_pred = torch.gather(keypoints_pred, dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.num_keypoints, 2))

        # For ONNX export (optional)
        if self.deploy_mode:
            return labels, boxes, scores, keypoints_pred

        # Handle COCO category remapping if needed (for keypoints as well, if necessary)
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco, kpts in zip(labels, boxes, scores, keypoints_pred):
            result = dict(labels=lab, boxes=box, scores=sco, keypoints=kpts)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', 'keypoints')  # Adding 'keypoints' to the IOU types
