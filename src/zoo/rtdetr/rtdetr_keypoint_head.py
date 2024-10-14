import torch
import torch.nn as nn
from typing import Union, Sequence, Tuple, List

__all__ = ['YOLOXPoseHeadModule']

from src.core import register


class ConvModule(nn.Module):
    """Convolutional Module with optional BatchNorm and Activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_cfg=None, act_cfg=None):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=norm_cfg is None)]
        
        if norm_cfg is not None:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if act_cfg is not None:
            layers.append(nn.SiLU(inplace=True)) if act_cfg.get('type') == 'SiLU' else layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        return self.block(x)


@register
class YOLOXPoseHeadModule(nn.Module):
    """YOLOXPose head module for one-stage human pose estimation."""
    def __init__(
        self,
        num_keypoints: int,
        in_channels: Union[int, Sequence],
        num_classes: int = 1,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        conv_bias: Union[bool, str] = 'auto',
        norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type='SiLU', inplace=True),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        self.conv_bias = conv_bias
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        elif isinstance(in_channels, list):
            in_channels = [int(c * widen_factor) for c in in_channels]  # Scale if it's a list of channels
        
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_cls_branch()
        self._init_reg_branch()
        self._init_pose_branch()

    def _init_cls_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()
        for i in range(len(self.featmap_strides)):
            stacked_convs = []
            for j in range(self.stacked_convs):
                chn = self.in_channels[i] if j == 0 and isinstance(self.in_channels, list) else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.conv_cls.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_cls.append(nn.Conv2d(self.feat_channels, self.num_classes, 1))

    def _init_reg_branch(self):
        """Initialize regression branch for all level feature maps."""
        self.conv_reg = nn.ModuleList()
        for i in range(len(self.featmap_strides)):
            stacked_convs = []
            for j in range(self.stacked_convs):
                chn = self.in_channels[i] if j == 0 and isinstance(self.in_channels, list) else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.conv_reg.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_bbox = nn.ModuleList()
        self.out_obj = nn.ModuleList()  # Added objectness output layer here
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(self.feat_channels, 4, 1))
            self.out_obj.append(nn.Conv2d(self.feat_channels, 1, 1))  # Objectness score

    def _init_pose_branch(self):
        """Initialize pose estimation branch for all level feature maps."""
        self.conv_pose = nn.ModuleList()
        for i in range(len(self.featmap_strides)):
            stacked_convs = []
            for j in range(self.stacked_convs * 2):
                in_chn = self.in_channels[i] if j == 0 and isinstance(self.in_channels, list) else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        in_chn,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        # output layers for keypoints
        self.out_kpt = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_kpt.append(nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(self.feat_channels, self.num_keypoints, 1))

    def forward(self, feature_maps: List[torch.Tensor]) -> Tuple[List]:
    # Assuming the input feature_maps are the 3 tensors you mentioned

        cls_scores, bbox_preds, objectnesses = [], [], []
        kpt_offsets, kpt_vis = [], []  # Initialize empty lists for keypoint offsets and visibility predictions

        for i in range(len(feature_maps)):
            # Process feature map for classification branch
            cls_feat = self.conv_cls[i](feature_maps[i])
            cls_scores.append(self.out_cls[i](cls_feat))

            # Process feature map for bbox regression branch
            reg_feat = self.conv_reg[i](feature_maps[i])
            bbox_preds.append(self.out_bbox[i](reg_feat))
            objectnesses.append(self.out_obj[i](reg_feat))

            # Use the same feature map to predict keypoint offsets and visibility
            pose_feat = self.conv_pose[i](feature_maps[i])
            kpt_offsets.append(self.out_kpt[i](pose_feat))  # Keypoint offset predictions
            kpt_vis.append(self.out_kpt_vis[i](pose_feat))  # Keypoint visibility predictions

        # Return predictions for classification, bbox, objectness, keypoints
        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis

