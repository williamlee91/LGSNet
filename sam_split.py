import os
import pickle
from get_sliding_windows import video_process
from construct_feature import construct_feature
import argparse
import pandas as pd
import numpy as np


def samm_train_test(ann_path, info_dir_split, info_dir_feature,
                    sample_interval,wind_length,window_sliding,
                    tem_feature_dir, spa_feature_dir, label_frequency, 
                    train_sample, is_train):

    if not os.path.exists(info_dir_split):
        os.makedirs( info_dir_split)
    # warming!!!!
    # 1280, 2000: 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,28,30,32,33,34,35,36,37,99 merge 31 and 23
    # 3000, 4500: 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,30,32,33,34,35,36,37,99 merge 31 and 23 and 28
    # 6000: 6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,30,32,33,34,35,36,37,99 merge 31 and 23 and 28 and 9 and 24
    sa_subject = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,28,30,32,33,34,35,36,37,99]
    
    sa_subject = [str(i).zfill(3)  for i in sa_subject]
    
    for i in range(len(sa_subject)):
        sa_subject_test = sa_subject[i]
        # separate training and testing samples
        ann_df = pd.read_csv(ann_path)
        # the data of training and test were kept separately
        ann_df_train = pd.read_csv(ann_path)
        ann_df_test = pd.read_csv(ann_path)
        for i in range(len(ann_df)):
            if str(ann_df.subject.values[i]).zfill(3) == sa_subject_test:
                ann_df_train.drop([i], inplace= True)
            else:
                ann_df_test.drop([i], inplace= True)
        
        # training window sliding
        gt_label, gt_box, gt_windows = video_process(ann_df_train, window_sliding, label_frequency, train_sample, neg=True)
        with open(os.path.join(info_dir_split, 'gt_label_{}.pkl'.format(sa_subject_test)), 'wb') as f:
            pickle.dump(gt_label, f)
        with open(os.path.join(info_dir_split, 'gt_box_{}.pkl'.format(sa_subject_test)), 'wb') as f:
            pickle.dump(gt_box, f)
        with open(os.path.join(info_dir_split, 'window_info_train_{}.log'.format(sa_subject_test)), 'w') as f:
            f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows)
        # save training dataset
        feature_process(sa_subject_test,
                        gt_label, gt_box, 
                        gt_windows,info_dir_feature,
                        sample_interval,wind_length,tem_feature_dir,
                        spa_feature_dir)

        # testing window sliding
        gt_label_test, gt_box_test, gt_windows_test = video_process(ann_df_test, window_sliding,label_frequency,train_sample, is_train=False)
        with open(os.path.join(info_dir_split, 'window_info_test_{}.log'.format(sa_subject_test)), 'w') as f:
            f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows_test)
        # save testing dataset
        feature_process(sa_subject_test,
                        gt_label_test, gt_box_test, 
                        gt_windows_test,info_dir_feature, 
                        sample_interval,wind_length,tem_feature_dir,
                        spa_feature_dir, False)


def feature_process(subject_test,
                    gt_label, gt_box, 
                    gt_windows,info_dir_feature, 
                    sample_interval,wind_length,tem_feature_dir,
                    spa_feature_dir, is_train=True):

    if is_train:
        sub_file_name = 'train'
    else:
        sub_file_name = 'test'

    print(sub_file_name, subject_test)
    
    save_feature_path = os.path.join(info_dir_feature, 'subject_{}'.format(subject_test), sub_file_name)
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
        
        mode = 'flow'
        feat_file = os.path.join(tem_feature_dir, vid_name + '-' + mode + '.npz')
        feat_tem = construct_feature(vid_name, feat_file, start_idx, end_idx, wind_length)
        mode = 'rgb'
        feat_file = os.path.join(spa_feature_dir, vid_name + '-' + mode + '.npz')
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
    parser.add_argument('--ann_path', type=str, default='/home/yww/1_spot/samm_annotation_merge_part_2000_L800_new2.csv')
    parser.add_argument('--info_dir_feature', type=str, default='/home/yww/1_spot/sa_subject_train_part_sample_2000_8_128_L800_new5_more/val_test')
    parser.add_argument('--info_dir_split', type=str, default='/home/yww/1_spot/sa_subject_train_part_sample_2000_8_128_L800_new5_more/subjects_split')
    parser.add_argument('--sample_interval', type=int, default=2)
    parser.add_argument('--wind_length', type=int, default=64)
    parser.add_argument('--window_sliding', type=int, default=128)
    parser.add_argument('--tem_feature_dir', type=str, default='/home/yww/1_spot/samm_merge_part_feature_sample_flow_2000_8')
    parser.add_argument('--spa_feature_dir', type=str, default='/home/yww/1_spot/samm_merge_part_feature_sample_img_2000_8')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--label_frequency', type=float, default=1.0)
    parser.add_argument('--train_sample', type=int, default=16)
    args = parser.parse_args()

    ann_path = args.ann_path
    info_dir_split = args.info_dir_split
    info_dir_feature =  args.info_dir_feature
    sample_interval =  args.sample_interval
    wind_length =  args.wind_length
    window_sliding =  args.window_sliding
    tem_feature_dir = args.tem_feature_dir
    spa_feature_dir =  args.spa_feature_dir
    label_frequency =  args.label_frequency
    train_sample =  args.train_sample
    is_train = args.is_train
    
    samm_train_test(ann_path, 
                    info_dir_split, 
                    info_dir_feature,
                    sample_interval,
                    wind_length,
                    window_sliding,
                    tem_feature_dir,
                    spa_feature_dir,
                    label_frequency,
                    train_sample,
                    is_train)