import torch
import numpy as np

from core.utils_ab import jaccard_with_anchors


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def default_box(temporal_len, stride, a_ratios):
    '''
    Given anchor num, scale and aspect_ratios
    generate default anchor boxes
    '''
    width_set = [stride * ratio for ratio in a_ratios]
    center_set = [0.5 * stride + stride * i for i in range(temporal_len)]
    width_default = list()
    center_default = list()

    for i in range(temporal_len):
        for j in range(len(a_ratios)):
            width_default.append(width_set[j])
            center_default.append(center_set[i])
    width_default = np.array(width_default)
    width_default = torch.from_numpy(width_default).type_as(dtype)
    center_default = np.array(center_default)
    center_default = torch.from_numpy(center_default).type_as(dtype)

    return width_default, center_default


def anchor_box_adjust(cfg, anchors, layer):
    '''
    Obtain detection box according to anchor box and regression (x, w)
    anchors: bs, ti*n_box, nclass+3
    '''
    temporal_len = cfg.MODEL.TEMPORAL_LENGTH[layer]
    stride = cfg.MODEL.TEMPORAL_STRIDE[layer]
    # dboxes_w, dboxes_x: ti*n_box.
    dboxes_w, dboxes_x = default_box(temporal_len, stride, cfg.MODEL.ASPECT_RATIOS)
    # temporal overlap
    anchors_rx = anchors[:, :, -2]
    anchors_rw = anchors[:, :, -1]

    anchors_x = anchors_rx * dboxes_w + dboxes_x
    anchors_w = torch.exp(anchors_rw) * dboxes_w
    anchors_class = anchors[:, :, :cfg.DATASET.NUM_CLASSES]

    return anchors_class, anchors_x, anchors_w


def match_anchor_gt(cfg, num_actions, anchor_x, anchor_w, glabel, gbbox, num_match):
    '''
    For each anchor box, calculate the matched ground truth box: box_x, box_w, match_score, match_label
    There are three points not perfect enough:
    (1) Do not use the ground truth overlap.
        When actions are interporated by sliding windows, cannot be aware
        We should multiply match_score with gt_overlap
    (2) The deeper layer can cover the shallow layer
        When an anchor match an action, the matching in deeper layer can modify this withoud any check
        We should use match_scores to determine which is more reliable. (How many cases like this?)
    (3) We do not constrain there should be one match for each ground truth
        The network is prone to match with easy anchor
    anchor_x, anchor_w: t_i*n_box
    g_label: n_action,
    gbbox: n_action, 3
    '''
    match_x = torch.zeros(num_match).type_as(dtype)
    match_w = torch.zeros(num_match).type_as(dtype)
    match_scores = torch.zeros(num_match).type_as(dtype)
    match_labels = torch.zeros(num_match).type_as(dtypel)

    for idx in range(num_actions):
        label = glabel[idx]
        # improve point 1
        box_min = gbbox[idx, 0]
        box_max = gbbox[idx, 1]

        box_x = (box_min + box_max) / 2
        box_w = box_max - box_min

        # predict
        anchors_min = anchor_x - anchor_w / 2
        anchors_max = anchor_x + anchor_w / 2

        jaccards = jaccard_with_anchors(anchors_min, anchors_max, box_min, box_max)

        mask = torch.gt(jaccards, match_scores)
        mask = mask & torch.gt(jaccards, cfg.TRAIN.FG_TH)

        # Update values using mask
        fmask = mask.type(torch.float32)
        match_x = fmask * box_x + (1 - fmask) * match_x
        match_w = fmask * box_w + (1 - fmask) * match_w

        imask = mask.type(torch.long)  # [80]  match_label.size(): [80, 21], label.size(): [21]
        ref_label = torch.ones(match_labels.size()).type_as(dtypel)
        ref_label = ref_label * label  # [80]
        match_labels = imask * ref_label + (1 - imask) * match_labels

        match_scores = torch.max(jaccards, match_scores)

    return match_x, match_w, match_labels, match_scores


def anchor_bboxes_encode(cfg, anchors, glabels, gbboxes, g_action_nums, layer):
    '''
    Produce matched ground truth with each adjusted anchors
    anchors: bs, ti*n_box, nclass+3
    g_action_nums: bs,
    '''

    temporal_length = cfg.MODEL.TEMPORAL_LENGTH[layer]
    stride = cfg.MODEL.TEMPORAL_STRIDE[layer]
    num_dbox = len(cfg.MODEL.ASPECT_RATIOS)

    # anchors_class: bs, ti*n_box, nclass. others: bs, ti*n_box
    anchors_rx = anchors[:, :, -2]
    anchors_rw = anchors[:, :, -1]
    anchors_class = anchors[:, :, :cfg.DATASET.NUM_CLASSES]
    # dboxes_w, dboxes_x: ti*n_box.
    dboxes_w, dboxes_x = default_box(temporal_length, stride, cfg.MODEL.ASPECT_RATIOS)

    batch_match_x = list()
    batch_match_w = list()
    batch_match_scores = list()
    batch_match_labels = list()

    num_data = int(g_action_nums.shape[0])
    for i in range(num_data):
        num_match = temporal_length * num_dbox

        action_num = g_action_nums[i]
        action_num = int(action_num.item())

        glabel = glabels[i, :action_num]  # num_action,
        gbbox = gbboxes[i, :action_num, :]  # num_action, 2

        match_x, match_w, match_labels, match_scores = match_anchor_gt(cfg, action_num, dboxes_x, dboxes_w, glabel, gbbox, num_match)

        match_x = torch.unsqueeze(match_x, dim=0)  # 1, ti*n_box
        batch_match_x.append(match_x)

        match_w = torch.unsqueeze(match_w, dim=0)
        batch_match_w.append(match_w)

        match_scores = torch.unsqueeze(match_scores, dim=0)
        batch_match_scores.append(match_scores)

        match_labels = torch.unsqueeze(match_labels, dim=0)
        batch_match_labels.append(match_labels)

    batch_match_x = torch.cat(batch_match_x, dim=0)
    batch_match_w = torch.cat(batch_match_w, dim=0)
    batch_match_scores = torch.cat(batch_match_scores, dim=0)
    batch_match_labels = torch.cat(batch_match_labels, dim=0)
    anchors_x = dboxes_x.expand(num_data, temporal_length * num_dbox)
    anchors_w = dboxes_w.expand(num_data, temporal_length * num_dbox)

    return batch_match_x, batch_match_w, batch_match_scores, batch_match_labels, \
           anchors_x, anchors_w, anchors_rx, anchors_rw, anchors_class
