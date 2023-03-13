import torch
import torch.nn as nn
import numpy as np

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()

def abs_smooth(x, enhance, beta=1. / 9):
    '''
    Conditional Soomth L1 loss
    Defined as:
        0.5*x^2 / bata       if abs(x) < beta
        abs(x) - 0.5*beta     if abs(x) >= beta
    '''
    absx = torch.abs(x)
    beta = torch.tensor(beta).type_as(dtype)
    minx = torch.min(absx, beta)
    loss = 0.5 * ((absx - beta) * minx + absx)
    # loss = torch.where(absx < beta, 0.5 * absx ** 2 / beta, absx - 0.5 * beta)
    loss = (enhance*loss).mean()
    return loss

def one_hot_embedding(labels, num_classes):
    y_tmp = torch.eye(num_classes, device=labels.device)  
    return y_tmp[labels]           


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=20, eps=1e-6):
        super(Focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, x, y):
        t = one_hot_embedding(y, 1 + self.num_classes)
        t = t[:, 1:]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        pt = pt.clamp(min=self.eps)  # avoid log(0)
        self.alpha = torch.tensor(self.alpha).cuda() 
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        loss = -(w * (1 - pt).pow(self.gamma) * torch.log(pt))
        return loss.sum()


def loss_function_ab(anchors_x, anchors_w, anchors_rx_ls, anchors_rw_ls, anchors_class,
                     match_x, match_w, match_scores, match_labels, cfg):
    '''
    calculate classification loss, localization loss and overlap_loss
    pmask, hmask and nmask are used to select training samples
    anchors_class: bs, sum_i(ti*n_box), nclass
    others: bs, sum_i(ti*n_box)
    '''
    batch_size = anchors_class.size(0)
    target_rx = (match_x - anchors_x) / anchors_w
    target_rw = torch.log(match_w / anchors_w)

    match_scores = match_scores.view(-1)
    pmask = match_scores > cfg.TRAIN.FG_TH
    nmask = match_scores < cfg.TRAIN.BG_TH

    # classification loss
    keep = (pmask.float() + nmask.float()) > 0
    anchors_class = anchors_class.view(-1, cfg.DATASET.NUM_CLASSES)[keep]
    match_labels = match_labels.view(-1)[keep]
    cls_loss_f = Focal_loss(alpha=cfg.TRAIN.ALPHA, num_classes=cfg.DATASET.NUM_CLASSES)
    cls_loss = cls_loss_f(anchors_class, match_labels) / (torch.sum(pmask)+ batch_size) # avoid no positive

    # localization loss
    if torch.sum(pmask) > 0:
        keep = pmask
        target_rx = target_rx.view(-1)[keep]
        target_rw = target_rw.view(-1)[keep]
        anchors_rx_ls = anchors_rx_ls.view(-1)[keep]
        anchors_rw_ls = torch.clamp(anchors_rw_ls.view(-1)[keep], max=5.0)
        target_loc = torch.stack(
            (match_x.view(-1)[keep]
            -0.5*anchors_w.view(-1)[keep]*torch.exp(target_rw), 
            match_x.view(-1)[keep]
            +0.5*anchors_w.view(-1)[keep]*torch.exp(target_rw)), 
            dim=-1
        )
        if cfg.TRAIN.ENHANCE_FLAG==1:
            duration_ab = target_loc[:,1] - target_loc[:,0]
            unenhance_dur = (duration_ab<cfg.TRAIN.SHORT_DURATION) + (duration_ab>cfg.TRAIN.LONG_DURATION)
            enhance_dur = (duration_ab>=cfg.TRAIN.SHORT_DURATION) * (duration_ab<=cfg.TRAIN.LONG_DURATION)
            enhance = cfg.TRAIN.ENHANCE_TIMES * torch.ones_like(duration_ab) * enhance_dur + torch.ones_like(duration_ab) * unenhance_dur
        # elif cfg.TRAIN.ENHANCE_FLAG==2:
        #     duration_ab = target_loc[:,1] - target_loc[:,0]
        #     unenhance_dur = (duration_ab > cfg.TRAIN.DURATION_MI) * (duration_ab < cfg.TRAIN.DURATION_LA)
        #     enhance_dur_short = duration_ab <= cfg.TRAIN.DURATION_LE
        #     enhance_dur_middle = (duration_ab > cfg.TRAIN.DURATION_LE) * (duration_ab <= cfg.TRAIN.DURATION_MI)
        #     enhance_dur_long = duration_ab >= cfg.TRAIN.DURATION_LA
        #     enhance = (cfg.TRAIN.LE_ENHANCE * torch.ones_like(duration_ab) * enhance_dur_short 
        #         + cfg.TRAIN.MI_ENHANCE * torch.ones_like(duration_ab) * enhance_dur_middle
        #         + torch.ones_like(duration_ab) * unenhance_dur
        #         + cfg.TRAIN.LA_ENHANCE * torch.ones_like(duration_ab) * enhance_dur_long)
        else:
            enhance = torch.ones_like(target_rx) 
        
        if cfg.TRAIN.ANCHOR_BASED_LOSS_FLAG==1:
            pred_loc = torch.stack(
                (anchors_rx_ls*anchors_w.view(-1)[keep] + anchors_x.view(-1)[keep] 
                - 0.5*anchors_w.view(-1)[keep]*torch.exp(anchors_rw_ls),
                anchors_rx_ls*anchors_w.view(-1)[keep] + anchors_x.view(-1)[keep] 
                + 0.5*anchors_w.view(-1)[keep]*torch.exp(anchors_rw_ls)), 
                dim=-1
            )
            loc_loss = iou_loss(pred_loc, target_loc, enhance)
        else:
            loc_factor = cfg.TRAIN.BETA_AB_LOC if cfg.TRAIN.BETA_SEPARATE else cfg.TRAIN.BETA_AB
            len_factor = cfg.TRAIN.BETA_AB_LEN if cfg.TRAIN.BETA_SEPARATE else cfg.TRAIN.BETA_AB
            loc_loss = (abs_smooth(target_rx - anchors_rx_ls, enhance, loc_factor) + abs_smooth(target_rw - anchors_rw_ls, enhance, len_factor))
        # rep_loss_f = RepulsionLoss(sigma=cfg.TRAIN.SIGMA, smooth=cfg.TRAIN.SMOOTH_THRESHOLD)
        # rep_loss = rep_loss_f(target_loc, pred_loc)
    else:
        loc_loss = torch.tensor(0.).type_as(cls_loss)
        rep_loss = torch.tensor(0.).type_as(cls_loss)
    # print('loss:', cls_loss.item(), loc_loss.item(), overlap_loss.item())
    loc_loss = loc_loss
    return cls_loss, loc_loss

def sel_fore_reg(cls_label_view, target_regs, pred_regs):
    '''
    Args:
        cls_label_view: bs*sum_t
        target_regs: bs, sum_t, 1
        pred_regs: bs, sum_t, 1
    Returns:
    '''
    sel_mask = cls_label_view >= 1.0
    target_regs_view = target_regs.view(-1)
    target_regs_sel = target_regs_view[sel_mask]
    pred_regs_view = pred_regs.view(-1)
    pred_regs_sel = pred_regs_view[sel_mask]

    return target_regs_sel, pred_regs_sel


def iou_loss(cfg, pred, target, enhance=None):
    inter_min = torch.max(pred[:, 0], target[:, 0])
    inter_max = torch.min(pred[:, 1], target[:, 1])
    inter_len = (inter_max - inter_min).clamp(min=1e-6)
    union_len = (pred[:, 1] - pred[:, 0]) + (target[:, 1] - target[:, 0]) - inter_len
    tious = inter_len / union_len
    if enhance is not None:
        loss = (enhance * (1 - tious)).mean()
    else:

        loss = (1 - tious).mean()
    return loss


def loss_function_af(cate_label, preds_cls, target_loc, pred_loc, allstri, reg_real, cfg):
    '''
    preds_cls: bs, t1+t2+..., n_class
    pred_regs_batch: bs, t1+t2+..., 2
    '''
    batch_size = preds_cls.size(0)
    cate_label_view = cate_label.view(-1)
    cate_label_view = cate_label_view.type_as(dtypel)
    preds_cls_view = preds_cls.view(-1, cfg.DATASET.NUM_CLASSES)
    pmask = (cate_label_view > 0).type_as(dtype)

    if torch.sum(pmask) > 0:
        # regression loss
        mask = pmask == 1.0
        proposals= cate_label_view[mask]
        if cfg.TRAIN.ENHANCE_FLAG==1:
            real_label = (reg_real * allstri).view(-1,2)[mask]
            real_duration = real_label[:,1] + real_label[:,0]
            unenhance_dur = (real_duration<cfg.TRAIN.SHORT_DURATION) + (real_duration>cfg.TRAIN.LONG_DURATION)
            enhance_dur = (real_duration>=cfg.TRAIN.SHORT_DURATION) * (real_duration<=cfg.TRAIN.LONG_DURATION)
            enhance = cfg.TRAIN.ENHANCE_TIMES * torch.ones_like(proposals) * enhance_dur + torch.ones_like(proposals) * unenhance_dur
        elif cfg.TRAIN.ENHANCE_FLAG==2:
            real_label = (reg_real * allstri).view(-1,2)[mask]
            real_duration = real_label[:,1] + real_label[:,0]
            unenhance_dur = (real_duration > cfg.TRAIN.DURATION_MI) * (real_duration < cfg.TRAIN.DURATION_LA)
            enhance_dur_short = real_duration <= cfg.TRAIN.DURATION_LE
            enhance_dur_middle = (real_duration > cfg.TRAIN.DURATION_LE) * (real_duration <= cfg.TRAIN.DURATION_MI)
            enhance_dur_long = real_duration >= cfg.TRAIN.DURATION_LA
            enhance = (cfg.TRAIN.LE_ENHANCE * torch.ones_like(proposals) * enhance_dur_short 
                + cfg.TRAIN.MI_ENHANCE * torch.ones_like(proposals) * enhance_dur_middle
                + torch.ones_like(proposals) * unenhance_dur
                + cfg.TRAIN.LA_ENHANCE * torch.ones_like(proposals) * enhance_dur_long)
        else:
            micro = proposals==2
            macro = proposals==1
            enhance = cfg.TRAIN.ENHANCE_TIMES * torch.ones_like(proposals) * micro + torch.ones_like(proposals) * macro
        pred_loc = pred_loc.view(-1, 2)[mask]
        target_loc = target_loc.view(-1, 2)[mask]
        reg_loss = iou_loss(cfg, pred_loc, target_loc, enhance)
        # # duration
        # dur_target = target_loc[:,1] - target_loc[:,0]
        # dur_pred = pred_loc[:,1] - pred_loc[:,0]
        # rep_loss_f = RepulsionLoss(sigma=cfg.TRAIN.SIGMA, smooth=cfg.TRAIN.SMOOTH_THRESHOLD)
        # rep_loss = rep_loss_f(target_loc, pred_loc)
    else:
        reg_loss = torch.tensor(0.).type_as(dtype)
        rep_loss = torch.tensor(0.).type_as(dtype)
    # cls loss
    cate_loss_f = Focal_loss(alpha=cfg.TRAIN.ALPHA, num_classes=cfg.DATASET.NUM_CLASSES)
    cate_loss = cate_loss_f(preds_cls_view, cate_label_view) / (torch.sum(pmask) + batch_size)  # avoid no positive
    reg_loss = reg_loss
    return cate_loss, reg_loss


class RepulsionLoss(nn.Module):
    def __init__(self, sigma=0.5 ,smooth=0.5, imp=1.0):
        super(RepulsionLoss, self).__init__()
        self.sigma = sigma
        self.smooth = smooth
        self.imp = imp
        
    def smooth_ln(self, x, smooth):
        return torch.where(
            x < smooth,
            -self.imp *torch.log(torch.clamp(x/smooth, min=1e-6)),
            (x - smooth) / (1 - smooth)
        )

    def IoG(self, box_a, box_b):
        inter_min = torch.max(box_a[:, 0], box_b[:, 0])
        inter_max = torch.min(box_a[:, 1], box_b[:, 1])
        I = torch.clamp(inter_max - inter_min, min=0)
        G = (box_a[:, 1] - box_a[:, 0])
        iog_v = torch.where(
            (I > 0) & (G > 0),
            I / G,
            torch.zeros(1, dtype=box_a.dtype, device=box_a.device)
        )
        return iog_v

    def forward(self, ground_data, loc_data):
        iog = self.IoG(ground_data, loc_data)
        iog_bias = self.smooth_ln(iog, self.smooth)
        loss = self.sigma * iog_bias.mean()       
        return loss


