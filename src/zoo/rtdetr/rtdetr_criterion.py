"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register


class KeypointMSELoss(nn.Module):
    """MSE loss for keypoint heatmaps, independent from mmpose."""

    def __init__(self, use_target_weight=False, skip_empty_channel=False, loss_weight=1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(self, output: torch.Tensor, target: torch.Tensor, target_weights=None, mask=None) -> torch.Tensor:
        """Calculate MSE loss."""
        mask = self._get_mask(target, target_weights, mask)
        if mask is None:
            loss = F.mse_loss(output, target)
        else:
            loss = F.mse_loss(output, target, reduction='none')
            loss = (loss * mask).mean()
        return loss * self.loss_weight

    def _get_mask(self, target: torch.Tensor, target_weights=None, mask=None) -> torch.Tensor:
        """Create mask based on target weights."""
        if target_weights is not None:
            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape + (1,) * ndim_pad)
            mask = target_weights if mask is None else mask * target_weights

        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)
            mask = _mask if mask is None else mask * _mask

        return mask


@register
class SetCriterion(nn.Module):
    """ This class computes the loss for RT-DETR, including keypoint loss.
    The process happens in two steps:
        1) Compute Hungarian assignment between ground truth boxes and the outputs of the model
        2) Supervise each pair of matched ground-truth / prediction (supervise class, box, and keypoints)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for the list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

        # Initialize KeypointMSELoss
        self.keypoint_loss_fn = KeypointMSELoss()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)."""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss, and the GIoU loss."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the MSE loss for keypoints."""
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_keypoints = outputs['pred_keypoints'][idx]  # Predicted keypoints
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # Ground truth

        # Calculate the keypoint MSE loss
        loss_keypoints = self.keypoint_loss_fn(src_keypoints, target_keypoints)

        return {'loss_keypoints': loss_keypoints}

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, i.e., the absolute error in the number of predicted non-empty boxes.
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Retrieve the appropriate loss function."""
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'keypoints': self.loss_keypoints,  # Added keypoint loss here
        }
        assert loss in loss_map, f'Do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def forward(self, outputs, targets):
    # Preprocess the outputs to match the expected format
        processed_outputs = {}

        
        
           
        
        if 'bbox_preds' in outputs and outputs['bbox_preds']:
            processed_outputs['pred_boxes'] = outputs['bbox_preds'][-1]  # Assume last item is the final prediction
        
        if 'kpt_offsets' in outputs and outputs['kpt_offsets']:
            processed_outputs['pred_keypoints'] = outputs['kpt_offsets'][-1]  # Assume last item is the final prediction
        
        # Now use processed_outputs instead of outputs
        outputs_without_aux = {k: v for k, v in processed_outputs.items() if 'aux' not in k}


        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)

        
        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        
        # Find the first tensor in the outputs dictionary to get its device
        first_tensor = next((v for v in outputs.values() if isinstance(v, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("No tensor found in outputs dictionary")
        
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=first_tensor.device)
        
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # ... (rest of the method remains the same)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, repeat with intermediate layers
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # Permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




