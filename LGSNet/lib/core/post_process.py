import pandas as pd
import os
import numpy as np
from core.nms import temporal_nms, temporal_nms_all, soft_nms, wbf_nms

def proposals_concate(cfg, df):
    start_time = np.array(df.xmin.values[:])
    end_time = np.array(df.xmax.values[:])
    scores = np.array(df.conf.values[:])
    duration = end_time - start_time
    order = scores.argsort()[::-1]
    keep = list()
    rstart = list()
    rend = list()
    rscore = list()
    while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):
        i = order[0]
        keep.append(i)
        tt1 = np.fabs(start_time[i] - end_time[order[1:]])
        tt2 = np.fabs(end_time[i] - start_time[order[1:]])
        # if selected proposals 
        if min(tt1) <=1 or min(tt2) <=1:

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


def final_result_process(out_df, epoch,subject, cfg, flag):
    '''
    flag:
    0: jointly consider out_df_ab and out_df_af
    1: only consider out_df_ab
    2: only consider out_df_af
    '''
    path_tmp = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, cfg.TEST.PREDICT_TXT_FILE)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    res_txt_file = os.path.join(path_tmp, 'test_' + str(epoch).zfill(2)+'.txt')
    if os.path.exists(res_txt_file):
        os.remove(res_txt_file)
    
    f = open(res_txt_file, 'a')

    if flag == 0:
        df_ab, df_af = out_df
        df_name = df_ab
    elif flag == 1:
        df_ab = out_df
        df_name = df_ab
    elif flag == 2:
        df_af = out_df
        df_name = df_af
    else:
        raise ValueError('flag should in {0, 1, 2}')

    video_name_list = list(set(df_name.video_name.values[:]))

    for video_name in video_name_list:
        if flag == 0:
            df_ab, df_af = out_df
            tmpdf_ab = df_ab[df_ab.video_name == video_name]
            tmpdf_af = df_af[df_af.video_name == video_name]
            tmpdf = pd.concat([tmpdf_ab, tmpdf_af], sort=True)
        elif flag == 1:
            tmpdf = df_ab[df_ab.video_name == video_name]
        else:
            tmpdf = df_af[df_af.video_name == video_name]

        type_set = list(set(tmpdf.cate_idx.values[:]))
        if epoch >= 15:
            if cfg.TEST.NMS_FALG==0:
                if cfg.TEST.NMS_ALL:
                    df_nms = temporal_nms_all(tmpdf, cfg)
                else:
                    df_nms = temporal_nms(tmpdf, cfg)
            elif cfg.TEST.NMS_FALG==1:
                df_nms = soft_nms(tmpdf, cfg)
            elif cfg.TEST.NMS_FALG==2:
                df_nms = wbf_nms(tmpdf, cfg)
        else:
            if cfg.TEST.NMS_ALL:
                df_nms = temporal_nms_all(tmpdf, cfg)
            else:
                df_nms = temporal_nms(tmpdf, cfg)
        # ensure there are most 200 proposals
        df_vid = df_nms.sort_values(by='score', ascending=False)
        for i in range(min(len(df_vid), cfg.TEST.TOP_K_RPOPOSAL)):
            start_time = df_vid.start.values[i]
            end_time = df_vid.end.values[i]
            try:
                label = df_vid.label.values[i]
                # length = end_time-start_time
                # if cfg.DATASET.DATASET_NAME =='samm':
                #     if label==1 and 101 < length:
                #         strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
                #     elif label==2 and 30 < length < 102:
                #         strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
                #     else:
                #         continue
                # else:
                #     if label==1 and 3 < length < 117:
                #         strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
                #     elif label==2 and 8 < length < 16:
                #         strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
                #     else:
                #         continue
                strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
                f.write(strout)
            except:
                strout = '%s\t%.3f\t%.3f\t%.4f\n' % (video_name, float(start_time), float(end_time), df_vid.score.values[i])
                f.write(strout)

    f.close()
