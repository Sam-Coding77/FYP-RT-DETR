import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src.core import register

@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""
    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_keypoints = weight_dict.get('cost_keypoints', 1)
        
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0 or self.cost_keypoints != 0, "All costs can't be 0"

    @torch.no_grad()
    
    def forward(self, outputs, targets):
        print("Debug: Entering forward method of HungarianMatcher")
        print(f"Debug: outputs keys: {outputs.keys()}")
        print(f"Debug: targets length: {len(targets)}")
        
        bs = outputs["pred_boxes"].shape[0]  # Batch size
        print(f"Debug: Batch size: {bs}")

        # Reshape pred_boxes to [batch_size, num_queries, 4]
        out_bbox = outputs["pred_boxes"].view(bs, -1, 4)
        print(f"Debug: Shape of reshaped out_bbox: {out_bbox.shape}")

        # Reshape pred_keypoints to [batch_size, num_queries, num_keypoints, 2]
        out_keypoints = outputs["pred_keypoints"].view(bs, -1, 17, 2)
        print(f"Debug: Shape of reshaped out_keypoints: {out_keypoints.shape}")

        # Process targets
        tgt_ids = [v["labels"].long() for v in targets]
        tgt_bbox = [v["boxes"] for v in targets]
        tgt_keypoints = [v.get("keypoints") for v in targets]

        for i, (ids, bbox, keypoints) in enumerate(zip(tgt_ids, tgt_bbox, tgt_keypoints)):
            print(f"Debug: Target {i} - labels: {ids.shape}, boxes: {bbox.shape}, " +
                  f"keypoints: {keypoints.shape if keypoints is not None else None}")

        # Process keypoints individually for each image
        tgt_keypoints = []
        for v in targets:
            if "keypoints" in v:
                tgt_keypoints.append(v["keypoints"])
            else:
                tgt_keypoints.append(None)  # No keypoints for this target image

        # Optional debug: check if any image is missing keypoints
        for i, kpts in enumerate(tgt_keypoints):
            if kpts is None:
                print(f"Image {i} has no keypoints.")
            else:
                print(f"Image {i} has keypoints of shape: {kpts.shape}")

        # Compute the L1 cost between boxes
        cost_bbox = [torch.cdist(out_bbox[i], tgt_box, p=1) for i, tgt_box in enumerate(tgt_bbox) if len(tgt_box) > 0]

        # Compute the giou cost between boxes
        cost_giou = []
        for i, tgt_box in enumerate(tgt_bbox):
            if len(tgt_box) > 0:
                pred_boxes_xyxy = box_cxcywh_to_xyxy(out_bbox[i])
                tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_box)
                
                # Clamp boxes to ensure they are valid
                pred_boxes_xyxy = torch.stack([
                    pred_boxes_xyxy[:, 0],
                    pred_boxes_xyxy[:, 1],
                    torch.max(pred_boxes_xyxy[:, 2], pred_boxes_xyxy[:, 0] + 1e-4),
                    torch.max(pred_boxes_xyxy[:, 3], pred_boxes_xyxy[:, 1] + 1e-4)
                ], dim=1)

                tgt_boxes_xyxy = torch.stack([
                    tgt_boxes_xyxy[:, 0],
                    tgt_boxes_xyxy[:, 1],
                    torch.max(tgt_boxes_xyxy[:, 2], tgt_boxes_xyxy[:, 0] + 1e-4),
                    torch.max(tgt_boxes_xyxy[:, 3], tgt_boxes_xyxy[:, 1] + 1e-4)
                ], dim=1)
                
                cost_giou_i = -generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
                cost_giou.append(cost_giou_i)

        # Compute the keypoint cost if applicable
        cost_keypoints = []
        matched_keypoints_pred = []
        matched_keypoints_target = []
        for i, tgt_kpt in enumerate(tgt_keypoints):
            if tgt_kpt is not None and len(tgt_kpt) > 0:
                tgt_kpt_xy = tgt_kpt[:, :, :2]  # Use only x and y coordinates
                
                out_kpt_reshaped = out_keypoints[i].view(-1, 17, 2)  # Reshape to [num_predictions, 17, 2]
                
                # Compute the L1 distance between predicted and target keypoints
                cost_keypoints_i = torch.cdist(out_kpt_reshaped.reshape(-1, 34), tgt_kpt_xy.reshape(-1, 34), p=1)
                
                cost_keypoints.append(cost_keypoints_i)
                matched_keypoints_pred.append(out_kpt_reshaped)
                matched_keypoints_target.append(tgt_kpt_xy)
            else:
                cost_keypoints.append(torch.zeros((out_keypoints[i].shape[0], 0), device=out_keypoints.device))
                matched_keypoints_pred.append(None)
                matched_keypoints_target.append(None)

        # Ensure all cost matrices have the same shape
        max_num_objects = max(max(cost.shape[1] for cost in cost_bbox), 1)
        C = []
        for i in range(bs):
            if i < len(cost_bbox) and cost_bbox[i].shape[1] > 0:
                cost_bbox_i = cost_bbox[i]
                cost_giou_i = cost_giou[i]
                cost_keypoints_i = cost_keypoints[i] if i < len(cost_keypoints) else torch.zeros_like(cost_bbox_i)
            else:
                print(f"Warning: No valid costs for image {i}")
                C.append(torch.empty((0, 0), device=out_bbox.device))
                continue

            C_i = self.cost_bbox * cost_bbox_i + self.cost_giou * cost_giou_i + self.cost_keypoints * cost_keypoints_i
            C.append(C_i)

        
        sizes = [len(v["boxes"]) for v in targets]
        
        indices = []
        for i, c in enumerate(C):
            print(f"Debug: Cost matrix for image {i}:")
            print(c)
            
            if c.shape[1] > 0:
                indices_i, indices_j = linear_sum_assignment(c.cpu().numpy())
                indices.append((torch.as_tensor(indices_i, dtype=torch.int64), 
                                torch.as_tensor(indices_j, dtype=torch.int64)))
            else:
                print(f"Warning: Empty cost matrix for image {i}. Skipping assignment.")
                indices.append((torch.tensor([], dtype=torch.int64), 
                                torch.tensor([], dtype=torch.int64)))

        print("Debug: Exiting forward method of HungarianMatcher")
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], matched_keypoints_pred, matched_keypoints_target
