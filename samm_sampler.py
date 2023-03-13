import os
import argparse
import random
import glob
import cv2
import copy
import math
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from extract_optical_flow import extract_flow

def label_new_all(ann_ori_path, new_ann_path, sw):
    ann = pd.read_csv(ann_ori_path)
    subject = list(set(ann.subject.values[:].tolist()))
    subject.sort()
    ori_pic_path = list(set(ann.video.values[:].tolist()))
    ori_pic_path.sort()
    
    new_subject = list()
    new_video_names = list()
    new_type = list()
    new_type_idx = list()
    new_start_frame = list()
    new_end_frame = list()
    new_frame_num = list()
    new_length = list()

    short_subject = list()
    short_video_names = list()
    short_type = list()
    short_type_idx = list()
    short_start_frame = list()
    short_end_frame = list()
    short_frame_num = list()
    short_length = list()

    subject_tmp = list()
    video_names_tmp = list()
    type_tmp = list()
    type_idx_tmp = list()
    start_frame_tmp = list()
    end_frame_tmp = list()
    length_tmp = list()

    short_length_all = 0
    short_count = 0
    for i in range(len(subject)):
        ann_df = ann[subject[i]==ann.subject.values[:]]
        one_video_path = list(set(ann_df['video'].values[:].tolist()))
        one_video_path = sorted(one_video_path, key = lambda x:int(x[-1:]))
        length_all = 0
        count = 0
        for j in range(len(one_video_path)):
            video_tmp = ann[one_video_path[j]==ann.video.values[:]]
            start_frame = video_tmp['startFrame'].values[:]
            end_frame = video_tmp['endFrame'].values[:]
            start_frame_tmp = start_frame_tmp + list(start_frame + length_all)
            end_frame_tmp = end_frame_tmp + list(end_frame + length_all)
            subject_tmp = subject_tmp + list(video_tmp['subject'].values[:])
            video_names_tmp = video_names_tmp + [val for val in [one_video_path[j][:-2]] for i in range(len(video_tmp))]
            type_tmp = type_tmp + list(video_tmp['type'].values[:])
            type_idx_tmp = type_idx_tmp + list(video_tmp['type_idx'].values[:])
            length_tmp = length_tmp + list(video_tmp['length'].values[:])
            length_all = length_all + list(video_tmp['frame_num'].values[:])[0]
            count = count + len(video_tmp)

        if  length_all > sw: # 512*12 = 6144 
            new_start_frame = new_start_frame + start_frame_tmp
            new_end_frame = new_end_frame + end_frame_tmp 
            new_subject = new_subject + subject_tmp
            new_video_names = new_video_names + video_names_tmp
            new_type = new_type + type_tmp
            new_type_idx = new_type_idx + type_idx_tmp
            new_length = new_length + length_tmp
            new_frame_num = new_frame_num + [val for val in [length_all] for i in range(count)]
        
        # find short sequences and merge them (23,24,28,31)
        # create new subject (99) including subject 23,24,28,31
        
        else:
            short_start_frame = short_start_frame + list(np.array(start_frame_tmp) + short_length_all)
            short_end_frame = short_end_frame + list(np.array(end_frame_tmp) + short_length_all)
            short_type = short_type + type_tmp
            short_type_idx = short_type_idx + type_idx_tmp
            short_length = short_length + length_tmp
            short_subject = short_subject + list([val for val in [99] for i in range(count)])
            short_video_names = short_video_names + [val for val in [one_video_path[j][:-4]+'99'] for i in range(count)]
            
            short_length_all = short_length_all + length_all
            short_count = short_count + len(ann_df)

        length_all = 0
        count = 0
        subject_tmp = list()
        video_names_tmp = list()
        type_tmp = list()
        type_idx_tmp = list()
        start_frame_tmp = list()
        end_frame_tmp = list()
        length_tmp = list()
    short_frame_num = short_frame_num + [val for val in [short_length_all] for i in range(short_count)]
    dic_inf = dict()
    dic_inf['subject'] = new_subject + short_subject
    dic_inf['video'] = new_video_names + short_video_names
    dic_inf['type'] = new_type + short_type
    dic_inf['type_idx'] = new_type_idx + short_type_idx
    dic_inf['startFrame'] = new_start_frame + short_start_frame
    dic_inf['endFrame'] = new_end_frame + short_end_frame 
    dic_inf['frame_num'] = new_frame_num + short_frame_num
    dic_inf['length'] = new_length + short_length

    df = pd.DataFrame(dic_inf, columns=['subject', 'video', 'type', 'type_idx', 'startFrame', 'endFrame', 'frame_num', 'length'])
    df.sort_values(['video', 'type_idx', 'startFrame'], inplace=True)
    df.to_csv(new_ann_path, encoding='utf-8', index=False)

    merge_subject = [i for i in subject if i not in new_subject]
    return merge_subject, ori_pic_path


def merge_verify(one_length, pname, an, sil_win, mode='video'):
    # verify whether there are repeat length 
    # count_ana = Counter(one_length)
    # for _, v in count_ana.items():
    #     if v > 1:
    #         print('Flase')
    #         print(sub)
    pname = list(np.array(pname)[np.argsort(one_length)])
    one_length.sort()
    merge_path = list()
    delete_tmp = 0
    for i in range(len(one_length)):
        if mode == 'video':
            tmp = an[pname[i]==an.video.values[:]].frame_num.values[:].tolist()[0]
        else:
            tmp = sum(list(set(an[pname[i]==an.subject.values[:]].frame_num.values[:].tolist())))
        if tmp > sil_win and (delete_tmp > sil_win or i==0):
            break
        else:
            merge_path.append(pname[i])
        delete_tmp = delete_tmp + tmp
    return merge_path


def merge_all(merge_path, ann_df, ann_new, mode='video'):
    sum_length_video = 0
    all_merge_index = list()
    for m in merge_path:
        ann_merge = ann_df[m==ann_df.video.values[:]] if mode=='video' else ann_new[m==ann_new.subject.values[:]]
        path_tmp = list(set(ann_merge['video'].values[:].tolist()))[0]
        ann_new.loc[ann_merge.index.to_list(), 'startFrame'] = ann_new.loc[ann_merge.index.to_list(), 'startFrame'] + sum_length_video
        ann_new.loc[ann_merge.index.to_list(), 'endFrame'] = ann_new.loc[ann_merge.index.to_list(), 'endFrame'] + sum_length_video 
        try:
            a = int(path_tmp[0][-2:])
            a = -2
        except:
            a = -1
        # replace path     
        ann_new.loc[ann_merge.index.to_list(), 'video'] = path_tmp[:a]+'99' if mode=='video' else os.path.join(os.path.dirname(path_tmp),'099_99')
        sum_length_video = sum_length_video + list(set(ann_merge['frame_num'].values[:].tolist()))[0]
        all_merge_index = all_merge_index + ann_merge.index.to_list()
    ann_new.loc[all_merge_index,'frame_num'] = sum_length_video # replace length 
    if mode != 'video':
        ann_new.loc[all_merge_index,'subject'] = 99

def label_new_part(ann_ori_path, new_ann_path, sw, nimgpath):
    ann = pd.read_csv(ann_ori_path)
    ann_new = copy.deepcopy(ann)
    subject = list(set(ann.subject.values[:].tolist()))
    subject.sort()
    ori_pic_path = list(set(ann.video.values[:].tolist()))
    ori_pic_path.sort()
    subject_long = copy.deepcopy(subject)
    
    # find the length of all videos of one subject is less than the sliding window
    subject_length_all = list()
    for j in range(len(subject)):
        ann_df_tmp = ann[subject[j] == ann.subject.values[:]]
        one_video_length_sub = list(set(ann_df_tmp['frame_num'].values[:].tolist()))
        # some length is repeated
        if sum(one_video_length_sub) in subject_length_all:
            subject_length_all.append(sum(one_video_length_sub) + 0.1)
            print("subject %d is repeated"% subject[j])
        else:
            subject_length_all.append(sum(one_video_length_sub))
    sla = copy.deepcopy(subject_length_all)
    # find subjects which shouled be merged
    merge_subject = merge_verify(sla, subject, ann, sw, 'subject')
    
    # merge short videos in each subject
    dict_merge_video = dict()
    for i in range(len(subject_long)):
        ann_df = ann[subject_long[i]==ann.subject.values[:]]
        one_video_path = list(set(ann_df['video'].values[:].tolist()))
        one_video_path = sorted(one_video_path, key = lambda x:int(x[-1:]))
        # verify that how many videos should be merged.
        one_video_length = list(set(ann_df['frame_num'].values[:].tolist()))
        one_video_path = list(set(ann_df['video'].values[:].tolist()))
        if len(one_video_path) != len(one_video_length):
            print("warming: some length of videos of subject %d is repeated!!!" %subject_long[i])
        one_video_length = list()
        for pa in one_video_path:
            tmp_l = ann_df[pa == ann_df['video'].values[:]].frame_num.values[:].tolist()[0]
            if tmp_l in one_video_length:
                one_video_length.append(tmp_l + 0.1)
            else:
                one_video_length.append(tmp_l)
        # verify how many videos are merged in each subject
        if subject_long[i] in merge_subject:
            print(" all videos of subject %d are merged"%subject_long[i])
            merge_video = list(np.array(one_video_path)[np.argsort(one_video_length)])
        else:    
            merge_video = merge_verify(one_video_length, one_video_path, ann_df, sw)
        # find merged videos which are named "0xx_99" 
        if merge_video:
            print(subject_long[i], " shorter videos should be merged")
            merge_all(merge_video, ann_df, ann_new, mode='video')
            dict_merge_video[subject_long[i]]= merge_video
    
    # merge videos from these subjects which should be merged
    merge_all(merge_subject, ann_df, ann_new, mode='subject')
    
    # transform old video paths into new paths
    base_file = os.path.basename(os.path.dirname(ori_pic_path[0]))
    last_file = os.path.basename(nimgpath)
    for v in range(len(ann_new)):
        ann_new.loc[v,'video'] = ann_new.loc[v,'video'].replace(base_file, last_file)
        ann_new.loc[v,'startFrame'] = math.ceil(ann_new.loc[v,'startFrame']/frequency)
        ann_new.loc[v,'endFrame'] = math.ceil(ann_new.loc[v,'endFrame']/frequency)
        ann_new.loc[v,'frame_num'] = math.ceil(ann_new.loc[v,'frame_num']/frequency)
        ann_new.loc[v,'length'] = ann_new.loc[v,'endFrame'] - ann_new.loc[v,'startFrame'] + 1
    ann_new.sort_values(['subject','video', 'type_idx', 'startFrame'], inplace=True)
    ann_new.to_csv(new_ann_path, encoding='utf-8', index=False)
    
    return dict_merge_video, merge_subject, ori_pic_path


def pic_merge(new_img_path, merge_subject, ori_pic_path, dict_merge_video=None):

    # the length of one video 
    length_tmp = 0
    # merge videos into a new video
    if dict_merge_video:
        for k, v in dict_merge_video.items():
            print("Videos of the subject %d are merging"%k)
            count = 0
            try:
                int(os.path.basename(v[0])[-1:])
                a = -1
            except:
                a = -2
            new_video_name = os.path.basename(v[0])[:a] + '99'
            save_img_path = os.path.join(new_img_path, new_video_name)
            if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)
            for vp in v:
                # delete selected the video path
                ori_pic_path.remove(vp)
                frames = glob.glob(os.path.join(vp, '*.jpg')) + glob.glob(os.path.join(vp, '*.png'))
                frames.sort()
                for i in tqdm(range(len(frames)), desc=new_video_name):
                    img_tmp = cv2.imread(frames[i])
                    pic_path = os.path.join(save_img_path, new_video_name + '_' +str(length_tmp + (count+i+1)).zfill(5)+'.jpg')
                    cv2.imwrite(pic_path, img_tmp)
                count = count + len(frames)
        print('videos of one subject have been merged')
    
    # copy remaining videos into a new file            
    for v_path in ori_pic_path:
        # all pics in one video
        frames = glob.glob(os.path.join(v_path, '*.jpg')) + glob.glob(os.path.join(v_path, '*.png'))
        frames.sort()
        save_repeat_path = v_path.replace(os.path.basename(os.path.dirname(v_path)), os.path.basename(new_img_path))
        if not os.path.exists(save_repeat_path):
            os.makedirs(save_repeat_path)
        for i in tqdm(range(len(frames)), desc=v_path):
            pic_tmp = cv2.imread(frames[i])
            new_tmp_path = frames[i].replace(os.path.basename(os.path.dirname(os.path.dirname(frames[i]))), os.path.basename(new_img_path))
            cv2.imwrite(new_tmp_path, pic_tmp)

    # merge videos of some subjects into a new subject
    if merge_subject:
        count_length_sub = 0
        # create new file of the merged subject
        save_merge_sub_path = os.path.join(new_img_path, '099_99')
        if not os.path.exists(save_merge_sub_path):
            os.makedirs(save_merge_sub_path)
        # copy pics from all videos of the subjects which will be merged
        for s in range(len(merge_subject)):
            print("a merged video of the subject %d are merging"%merge_subject[s])
            # find videos which will be merged into the subject "99"
            video_name = os.path.join(new_img_path, str(merge_subject[s]).zfill(3) + '_99')
            frames = glob.glob(os.path.join(video_name, '*.jpg')) + glob.glob(os.path.join(video_name, '*.png'))
            frames.sort()
            for i in tqdm(range(len(frames)), desc=video_name):
                pic_tmp_sub = cv2.imread(frames[i])
                new_pic_sub_path = os.path.join(save_merge_sub_path, str(i+count_length_sub).zfill(5)+'.jpg')
                cv2.imwrite(new_pic_sub_path, pic_tmp_sub)
            shutil.rmtree(video_name)
            count_length_sub = count_length_sub + len(frames)
    print('merging is finished!!!!!!!!!!')


def main(new_img_path, flow_path, save_path_img, save_path_flow, frequency, ann_ori_path, new_ann_path, sw, sample=False):
    
    # ann merge
    # merge_subject, ori_pic_path = label_new_all(ann_ori_path, new_ann_path, sw)
    dict_merge_video, merge_subject, ori_pic_path = label_new_part(ann_ori_path, new_ann_path, sw, new_img_path)
    
    # pic merge
    pic_merge(new_img_path, merge_subject, ori_pic_path, dict_merge_video)
    
    # down-sample the origin dataset
    if sample:
        all_video_merge_tmp = os.listdir(new_img_path)
        for m_path in tqdm(all_video_merge_tmp):
            # all pics in one video
            frames = glob.glob(os.path.join(new_img_path, m_path, '*.jpg')) + glob.glob(os.path.join(new_img_path, m_path, '*.png'))
            frames.sort()

            number_frame_video = len(frames)
            # compression 10x or 5x or 1x
            new_frames = list()
            frequency_para = [6, 7, 7]
            end = 0
            for j in range(math.ceil(number_frame_video/frequency)):
                end_tmp = number_frame_video-1 if end + frequency_para[j%len(frequency_para)] >= number_frame_video else end + frequency_para[j%len(frequency_para)]
                new_frames.append(frames[end])
                end = end_tmp
            # for j in range(0, number_frame_video, frequency):
            #     if j+frequency >= number_frame_video:
            #         end_tmp = number_frame_video-1
            #     else:
            #         end_tmp = j+frequency
            #     tmp = random.randint(j, end_tmp)
            #     new_frames.append(frames[tmp])

            # save all pics and flow into samoling fold with new names
            pre_pic_tmp = m_path.split('/')[-1]
            pic_tmp_file = os.path.join(save_path_img, pre_pic_tmp)
            if not os.path.exists(pic_tmp_file):
                os.makedirs(pic_tmp_file)
        
            for k in range(len(new_frames)):
                sub_tmp = str(k+1).zfill(4) + '.jpg'
                pic = cv2.imread(new_frames[k])
                pic_path = os.path.join(pic_tmp_file, sub_tmp)
                cv2.imwrite(pic_path, pic)
 
        # generate optical flow from sampled images
        extract_flow(save_path_img, save_path_flow, dataset='samm')
    else:
        # generate optical flow
        extract_flow(new_img_path, flow_path, dataset='samm')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_img_path', type=str, default='/home/yww/LGSNET/samm_merge_img_1600')
    parser.add_argument('--flow_path', type=str, default='/home/yww/LGSNET/samm_merge_flow_1600')
    parser.add_argument('--save_path_img', type=str, default='/home/yww/LGSNET/samm_img_merge_part_sample')
    parser.add_argument('--save_path_flow', type=str, default='/home/yww/LGSNET/samm_flow_merge_part_sample')
    parser.add_argument('--frequency', type=float, default=6.67)
    parser.add_argument('--ann_ori', type=str, default='/home/yww/LGSNET/samm_annotation.csv')
    parser.add_argument('--new_ann_path', type=str, default='/home/yww/LGSNET/samm_annotation_sample.csv')
    parser.add_argument('--is_sample', type=str, default=True)
    parser.add_argument('--sliding_window', type=int, default=2000, help='if the model is all, sw=6144')
    
    
    args = parser.parse_args()
    flow_path = args.flow_path
    save_path_img = args.save_path_img
    save_path_flow = args.save_path_flow
    frequency = args.frequency
    ann_ori_path = args.ann_ori
    new_ann_path = args.new_ann_path
    new_img_path = args.new_img_path
    is_sample = args.is_sample
    sw = args.sliding_window
    main(new_img_path, flow_path, save_path_img, save_path_flow, frequency, ann_ori_path, new_ann_path, sw, is_sample)