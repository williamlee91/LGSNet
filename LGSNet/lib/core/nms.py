import numpy as np
import pandas as pd

from core.utils_ab import tiou


def temporal_nms(df, cfg):
    '''
    temporal nms
    I should understand this process
    '''

    type_set = list(set(df.cate_idx.values[:]))
    # type_set.sort()

    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    # attention: for THUMOS, a sliding window may contain actions from different class
    for t in type_set:
        label = t
        tmp_df = df[df.cate_idx == t]

        start_time = np.array(tmp_df.xmin.values[:])
        end_time = np.array(tmp_df.xmax.values[:])
        scores = np.array(tmp_df.conf.values[:])

        duration = end_time - start_time
        order = scores.argsort()[::-1]

        keep = list()
        while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):
            i = order[0]
            keep.append(i)
            tt1 = np.maximum(start_time[i], start_time[order[1:]])
            tt2 = np.minimum(end_time[i], end_time[order[1:]])
            intersection = tt2 - tt1
            union = (duration[i] + duration[order[1:]] - intersection).astype(float)
            iou = intersection / union

            inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
            order = order[inds + 1]

        # record the result
        for idx in keep:
            rlabel.append(label)
            rstart.append(float(start_time[idx]))
            rend.append(float(end_time[idx]))
            rscore.append(scores[idx])

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df


def temporal_nms_all(df, cfg):
    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    # attention: for THUMOS, a sliding window may contain actions from different class
    tmp_df = df

    start_time = np.array(tmp_df.xmin.values[:])
    end_time = np.array(tmp_df.xmax.values[:])
    scores = np.array(tmp_df.conf.values[:])

    duration = end_time - start_time
    order = scores.argsort()[::-1]

    keep = list()
    while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(start_time[i], start_time[order[1:]])
        tt2 = np.minimum(end_time[i], end_time[order[1:]])
        intersection = tt2 - tt1
        union = (duration[i] + duration[order[1:]] - intersection).astype(float)
        iou = intersection / union

        inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
        order = order[inds + 1]
    # record the result
    for idx in keep:
        rstart.append(float(start_time[idx]))
        rend.append(float(end_time[idx]))
        rscore.append(scores[idx])

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    return new_df



def soft_nms(df, idx_name, cfg):
    df = df.sort_values(by='score', ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.conf.values[:])
    tcls_type = list(df.cls_type.values[:])
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    while len(tscore) > 0 and len(rscore) <= cfg.TEST.TOP_K_RPOPOSAL:
        max_idx = np.argmax(tscore)
        tmp_width = tend[max_idx] - tstart[max_idx]
        iou = tiou(tstart[max_idx], tend[max_idx], tmp_width, np.array(tstart), np.array(tend))
        iou_exp = np.exp(-np.square(iou) / cfg.TEST.SOFT_NMS_ALPHA)
        for idx in range(len(tscore)):
            if idx != max_idx:
                tmp_iou = iou[idx]
                threshold = cfg.TEST.SOFT_NMS_LOW_TH + (cfg.TEST.SOFT_NMS_HIGH_TH - cfg.TEST.SOFT_NMS_LOW_TH) * tmp_width
                if tmp_iou > threshold:
                    tscore[idx] = tscore[idx] * iou_exp[idx]
        rstart.append(tstart[max_idx])
        rend.append(tend[max_idx])
        rscore.append(tscore[max_idx])
        # video class label
        cls_type = tcls_type[max_idx]
        label = idx_name[cls_type]
        rlabel.append(label)

        tstart.pop(max_idx)
        tend.pop(max_idx)
        tscore.pop(max_idx)
        tcls_type.pop(max_idx)

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df



def wbf_nms(df, cfg):
    # adjust conf for each pred
    iou_thr = cfg.TEST.WBF_NMS_TH
    skip_box_thr = cfg.TEST.WBF_SKIP_TH
    df = df.sort_values(by='conf', ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    # 3 dimensions: models_number, model_preds, 2
    # expect to normlize boxes to [0,1]
    boxes_list = [[list(i) for i in list(zip(tstart, tend))]]
    conf_list = [list(df.conf.values[:])]
    # labels_list = [list(df.cate_idx.values[:])]
    labels_list = [np.ones((len(tstart),), dtype=int).tolist()]
    boxes, scores, _ = weighted_boxes_fusion(cfg, boxes_list, conf_list, labels_list, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    new_df = pd.DataFrame()
    new_df['start'] = list(boxes[:,0])
    new_df['end'] = list(boxes[:,1])
    new_df['score'] = list(scores)
    return new_df


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    # each model
    for t in range(len(boxes)):
        # all pred_loc in the each model
        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1])]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]
    return new_boxes


def get_weighted_box(cfg, boxes, conf_type='avg'):
    box = np.zeros(4, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        # bconf_tmp = b[1] 
        # box[2:] += (b[1] * b[2:])
        bconf_tmp = np.power(b[1], cfg.TEST.WBF_REWEIGHT_FACTOR)
        box[2:] += (bconf_tmp* b[2:])
        conf += bconf_tmp
        conf_list.append(bconf_tmp)    
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def iou_temporal(box, nbox):
    inter_xmin = np.maximum(box[0], nbox[0])
    inter_xmax = np.minimum(box[1], nbox[1])
    inter_len = np.maximum(inter_xmax-inter_xmin, 0.)
    union_len = (box[1]-box[0]) + (nbox[1]-nbox[0]) - inter_len
    tiou = np.divide(inter_len, union_len)
    return tiou


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = iou_temporal(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(cfg, boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(cfg, new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

