import os
import pickle
from get_sliding_windows import video_process
from construct_feature import construct_feature
import argparse
import pandas as pd
import numpy as np

def cas_train_test(ann_path, info_dir_split, info_dir_feature,
                    sample_interval,wind_length,window_sliding,
                    tem_feature_dir, spa_feature_dir,label_frequency,
                    is_train):

    if not os.path.exists(info_dir_split):
        os.makedirs( info_dir_split)
    ca_subject = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,39,40,41,42,77,138,139,140,142,143,144,145,146,147,148,\
            149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,165,166,167,168,169,170,171,172,173,\
            174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,192,193,194,195,196,197,198,\
            200,201,202,203,204,206,207,208,209,210,212,213,214,215,216,217,218]
    ca_subject = [str(i).zfill(3)  for i in ca_subject]
    for i in range(len(ca_subject)):
        ca_subject_test  = ca_subject[i]
        ann_df = pd.read_csv(ann_path)
        ann_df_train = pd.read_csv(ann_path)
        ann_df_test = pd.read_csv(ann_path)
        for i in range(len(ann_df)):
            if str(ann_df.subject.values[i]).zfill(3) == ca_subject_test:
                ann_df_train.drop([i], inplace= True)
            else:
                ann_df_test.drop([i], inplace= True)
        
        # training window sliding
        gt_label, gt_box, gt_windows = video_process(ann_df_train, window_sliding, label_frequency, neg=True)
        with open(os.path.join(info_dir_split, 'gt_label_{}.pkl'.format(ca_subject_test)), 'wb') as f:
            pickle.dump(gt_label, f)
        with open(os.path.join(info_dir_split, 'gt_box_{}.pkl'.format(ca_subject_test)), 'wb') as f:
            pickle.dump(gt_box, f)
        with open(os.path.join(info_dir_split, 'window_info_train_{}.log'.format(ca_subject_test)), 'w') as f:
            f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows)
        
        feature_process(ca_subject_test,
                        gt_label, gt_box, 
                        gt_windows,info_dir_feature,
                        sample_interval,wind_length,tem_feature_dir,
                        spa_feature_dir)

        # testing window sliding
        gt_label_test, gt_box_test, gt_windows_test = video_process(ann_df_test, window_sliding, label_frequency, is_train=False)
        with open(os.path.join(info_dir_split, 'window_info_test_{}.log'.format(ca_subject_test)), 'w') as f:
            f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows_test)
        feature_process(ca_subject_test,
                        gt_label_test, gt_box_test, 
                        gt_windows_test,info_dir_feature, 
                        sample_interval,wind_length,tem_feature_dir,
                        spa_feature_dir, False)


def feature_process(ca_subject_test,
                    gt_label, gt_box, 
                    gt_windows,info_dir_feature, 
                    sample_interval,wind_length,tem_feature_dir,
                    spa_feature_dir, is_train=True):

    if is_train:
        sub_file_name = 'train'
    else:
        sub_file_name = 'test'

    print(sub_file_name, ca_subject_test)
    
    save_feature_path = os.path.join(info_dir_feature, ca_subject_test, sub_file_name)
    if not os.path.exists(save_feature_path):
        os.makedirs(save_feature_path)
    
    for iord, line in enumerate(gt_windows):
        begin_frame, vid_path = line[0], line[1]
        vid_name = vid_path.split('/')[-2] + '_' + vid_path.split('/')[-1]
        last_save_file = os.path.join(save_feature_path, vid_name + '_' + str(begin_frame).zfill(5) + '.npz')
        
        if is_train:
            info = gt_box[iord]
            label = gt_label[iord]
        else:
            info = None
            label = None

        start_idx = int(int(begin_frame) / sample_interval)
        end_idx = start_idx + wind_length

        sub_name = vid_path.split('/')[-2]
        mode = 'flow'
        feat_file = os.path.join(tem_feature_dir, sub_name, vid_path.split('/')[-1] + '-' + mode + '.npz')
        feat_tem = construct_feature(vid_name, feat_file, start_idx, end_idx, wind_length)
        mode = 'rgb'
        feat_file = os.path.join(spa_feature_dir, sub_name, vid_path.split('/')[-1] + '-' + mode + '.npz')
        feat_spa = construct_feature(vid_name, feat_file, start_idx, end_idx, wind_length)

        np.savez(last_save_file,
                vid_name=vid_name,
                begin_frame=int(begin_frame),
                action=info,
                class_label=label,
                feat_tem=feat_tem,
                feat_spa=feat_spa)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default='/home/yww/1_spot/cas3_annotation_full_me_reduce.csv')
    parser.add_argument('--info_dir_feature', type=str, default='/home/yww/1_spot/cas3_3pos_new/val_test')
    parser.add_argument('--info_dir_split', type=str, default='/home/yww/1_spot/cas3_3pos_new/subjects_split')
    parser.add_argument('--sample_interval', type=int, default=2)
    parser.add_argument('--wind_length', type=int, default=64)
    parser.add_argument('--window_sliding', type=int, default=128)
    parser.add_argument('--tem_feature_dir', type=str, default='/home/yww/CAS^3/new_data_flow_feature')
    parser.add_argument('--spa_feature_dir', type=str, default='/home/yww/CAS^3/new_data_full_reduce_feature')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--label_frequency', type=int, default=1)
    args = parser.parse_args()
    cas_train_test(args.ann_path, 
                    args.info_dir_split, 
                    args.info_dir_feature,
                    args.sample_interval,
                    args.wind_length,
                    args.window_sliding,
                    args.tem_feature_dir,
                    args.spa_feature_dir,
                    args.label_frequency,
                    args.is_train)