import math
import numpy as np
import torch
from functools import partial

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()
INF = 1e8


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def get_points_single(tem_length, stride):
    stride_l = math.floor(stride / 2)
    points = [i * stride + stride_l for i in range(tem_length)]
    return points


def get_points(cfg):
    points = list()
    for layer in range(cfg.MODEL.NUM_LAYERS):
        tem_length = cfg.MODEL.TEMPORAL_LENGTH[layer]
        stride = cfg.MODEL.TEMPORAL_STRIDE[layer]
        layer_points = get_points_single(tem_length, stride)
        points.append(torch.tensor(layer_points).type_as(dtype))

    return points


def get_targets_af(cfg, gt_bboxes, gt_labels, action_nums):
    """

    Args:
        cfg:
        gt_bboxes: shape: bs, num_max_gt, 3
        gt_labels: bs, num_max_gt
        action_nums: bs,

    Returns:

    """
    points = get_points(cfg)
    regress_ranges = cfg.TRAIN.REGRESS_RANGES
    assert len(points) == len(regress_ranges)
    num_levels = len(points) # 4
    expanded_regress_ranges = [
        points[i].new_tensor(regress_ranges[i])[None].repeat( # [None] is used to add "[]" outsides 
            points[i].size(0), 1) for i in range(num_levels)  # n_point, 2, according to every elements of points # 16, 8, 4, 2
    ]
    concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
    concat_points = torch.cat(points)  # 16+8+4+2

    num_points = [p.size(0) for p in points]

    labels_list, bbox_targets_list = multi_apply(
        get_target_single,
        gt_bboxes,
        gt_labels,
        action_nums,
        points=concat_points,
        regress_ranges=concat_regress_ranges,
        num_points_per_lvl=num_points,
        cfg=cfg)
    
    labels = torch.stack(labels_list)
    bbox_targets = torch.stack(bbox_targets_list)  # bs, sum_i(t_i), 2
    if cfg.MODEL.NORM_ON_BBOX:
        strides = [
            points[i].new_tensor(cfg.MODEL.TEMPORAL_STRIDE[i]).expand(
                points[i].size(0)) for i in range(num_levels)  # n_point
        ]
        strides = torch.cat(strides)  # sum_i(t_i),
        assert strides.size(0) == bbox_targets.size(1)
        bbox_targets = bbox_targets / strides[None, :, None].expand_as(bbox_targets)

    all_strides = strides[None, :, None].expand_as(bbox_targets)
    return labels, bbox_targets, all_strides


def get_target_single(gt_bboxes, gt_labels, action_nums, points, regress_ranges,
                      num_points_per_lvl, cfg):
    """

    Args:
        gt_bboxes: shape: num_max_gt, 3
        gt_labels: shape: num_max_gt,
        action_nums: shape: 1,
        points:  n_point_all_levels,
        regress_ranges: n_point_all_levels, 2
        num_points_per_lvl: list

    Returns:

    """
    num_points = points.size(0)
    gt_bboxes = gt_bboxes[:action_nums, :]
    gt_labels = gt_labels[:action_nums]

    # gt_bboxes = gt_bboxes / 8  # match the max coordinate of point, which is 64, i.e. window_size / 8

    if action_nums == 0:
        return gt_labels.new_full((num_points,), 0), gt_bboxes.new_zeros((num_points, 2))

    gt_length = gt_bboxes[:, 1] - gt_bboxes[:, 0]
    gt_length = gt_length[None].repeat(num_points, 1)  # Could not use expand!!!!

    regress_ranges = regress_ranges[:, None, :].expand(num_points, action_nums, 2)
    gt_bboxes = gt_bboxes[None].expand(num_points, action_nums, 2) # expand as points
    points = points[:, None].expand(num_points, action_nums)

    left = points - gt_bboxes[..., 0]
    right = gt_bboxes[..., 1] - points
    bbox_targets = torch.stack((left, right), -1)  # n_point, n_gt, 2

    if cfg.TRAIN.CENTER_SAMPLING:
        radius = cfg.TRAIN.CENTER_SAMPLE_RADIUS
        centers = (gt_bboxes[..., 0] + gt_bboxes[..., 1]) / 2  # n_point, n_gt
        center_gts = torch.zeros_like(gt_bboxes)
        stride = centers.new_zeros(centers.shape)

        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = cfg.MODEL.TEMPORAL_STRIDE[lvl_idx] * radius[lvl_idx]
            lvl_begin = lvl_end

        center_mins = centers - stride
        center_maxs = centers + stride
        center_gts[..., 0] = torch.where(center_mins > gt_bboxes[..., 0],
                                         center_mins, gt_bboxes[..., 0])
        center_gts[..., 1] = torch.where(center_maxs > gt_bboxes[..., 1],
                                         gt_bboxes[..., 1], center_maxs)
        center_left = points - center_gts[..., 0]
        center_right = center_gts[..., 1] - points
        center_bbox = torch.stack((center_left, center_right), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
    else:
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

    max_regress_distance = bbox_targets.max(-1)[0]
    inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance < regress_ranges[..., 1]))
    
    gt_length[(inside_gt_bbox_mask & inside_regress_range) == 0] = INF
    min_length, min_length_inds = gt_length.min(dim=1)  # select the label with shortest length 
    
    labels = gt_labels[min_length_inds]
    labels[min_length == INF] = 0  # BG
    bbox_targets = bbox_targets[range(num_points), min_length_inds]
    bbox_targets[min_length == INF, :] = 0
    return labels, bbox_targets
