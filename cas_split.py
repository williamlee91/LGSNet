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
    
    ca_subject = [15,16,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,40]
    ca_label = [i for i in range(1, len(ca_subject)+1)]
    ca_subject = ['s'+ str(i)  for i in ca_subject]
    
    for i in range(len(ca_subject)):
        ca_subject_test = ca_subject[i]
        ca_lebel_test =  ca_label[i]
        ann_df = pd.read_csv(ann_path)
        ann_df_train = pd.read_csv(ann_path)
        ann_df_test = pd.read_csv(ann_path)
        for i in range(len(ann_df)):
            if int(ann_df.subject.values[i]) == ca_lebel_test:
                ann_df_train.drop([i], inplace= True)
            else:
                ann_df_test.drop([i], inplace= True)
        
        # training window sliding
        gt_label, gt_box, gt_windows = video_process(ann_df_train, window_sliding, label_frequency, neg=True)
        # len_tmp = 0
        # for i in range(len(list(gt_label))):
        #     if len(list(gt_label[i]))>len_tmp:
        #         len_tmp = len(list(gt_label[i]))
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
    
    save_feature_path = os.path.join(info_dir_feature, 'subject_{}'.format(ca_subject_test), sub_file_name)
    if not os.path.exists(save_feature_path):
        os.makedirs(save_feature_path)
    
    for iord, line in enumerate(gt_windows):
        begin_frame, vid_name = line[0], line[1]
        vid_name = vid_name.split('/')[-1]
        last_save_file =os.path.join(save_feature_path, vid_name + '_' + str(begin_frame).zfill(5) + '.npz')
        
        if is_train:
            info = gt_box[iord]
            label = gt_label[iord]
        else:
            info = None
            label = None

        start_idx = int(int(begin_frame) / sample_interval)
        end_idx = start_idx + wind_length
        
        sub_name = 's' + vid_name.split('_')[0]
        mode = 'flow'
        feat_file = os.path.join(tem_feature_dir, sub_name, vid_name + '-' + mode + '.npz')
        feat_tem = construct_feature(vid_name, feat_file, start_idx, end_idx, wind_length)
        mode = 'rgb'
        feat_file = os.path.join(spa_feature_dir, sub_name, vid_name + '-' + mode + '.npz')
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
    parser.add_argument('--ann_path', type=str, default='/home/yww/1_spot/casme2_annotation_357.csv')
    parser.add_argument('--info_dir_feature', type=str, default='/home/yww/1_spot/ca_subject_train_ori_new_3pos_357/val_test')
    parser.add_argument('--info_dir_split', type=str, default='/home/yww/1_spot/ca_subject_train_ori_new_3pos_357/subjects_split')
    parser.add_argument('--sample_interval', type=int, default=2)
    parser.add_argument('--wind_length', type=int, default=64)
    parser.add_argument('--window_sliding', type=int, default=128)
    parser.add_argument('--tem_feature_dir', type=str, default='/home/yww/1_spot/CA_flow_feature')
    parser.add_argument('--spa_feature_dir', type=str, default='/home/yww/1_spot/CA_rgb_feature')
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