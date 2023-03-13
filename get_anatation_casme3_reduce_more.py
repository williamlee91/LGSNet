import pandas as pd
import os
import argparse
import math
import numpy
import cv2

def process_one_video(fpath, svideopath, sss, before, after, start=0, end=0):
    # the basenames of all pics in one video
    all_pic = os.listdir(fpath) 
    print(fpath)
    all_pic = [int(str(i).split('.')[0]) for i in all_pic]
    all_pic.sort()
    # verify the squecne is from 0 
    # the name of first frame are used for calculate the number of the missing frames in the strating of one video 
    firstname = all_pic[0]
    if firstname == 0:
        firstname == -1
        print(fpath)
    # find the indexes of missing frames
    all_pic_diff = numpy.diff(numpy.array(all_pic))
    oom_index = numpy.argwhere(all_pic_diff >= 2).squeeze(-1)
    plunge_num = 0 # the number of plunging pics
    if oom_index.any() and fpath.split('yww')[1] not in sss:
        for k in oom_index.tolist():
            # devide the path of '/home/yww/CAS^3/part_A/data/part_A/spNO.x/x/color/xxx' into '/home/' and 'CAS^3/part_A/data/part_A/spNO.x/x/color/xxx'
            sss.append(fpath.split('yww')[1]) 
            before.append(all_pic[k])
            after.append(all_pic[k+1])
            # copy the first frame of missing squences as the missing frame in new data folder
            pic_tmp = cv2.imread(os.path.join(fpath, str(all_pic[k])+'.jpg'))
            for np in range (1, all_pic[k+1] - all_pic[k]):
                # print(str(all_pic[k]+np-(firstname-1))+'.jpg')
                cv2.imwrite(os.path.join(svideopath, str(all_pic[k]+np-(firstname-1))+'.jpg'), pic_tmp)    
                plunge_num = plunge_num + 1    
            # test missing frames in one GT ?
            if start <= all_pic[k]< end  and start< all_pic[k+1] <= end: 
                print("error gt!!", fpath, start, end)
    
    # copy remaining pics into the new data folder
    if len(os.listdir(svideopath)) < len(all_pic) + plunge_num:
        for p in range(len(all_pic)):
            pic_tmp = cv2.imread(os.path.join(fpath, str(all_pic[p])+'.jpg'))
            cv2.imwrite(os.path.join(svideopath, str(all_pic[p]- firstname + 1)+'.jpg'), pic_tmp)
            
    return sss, before, after, firstname


def main(e_path, v_path, l_path, extract_full_data, delete_long):
    
    df_sheet = pd.read_excel(e_path, header=0)
    subject = list()
    video_names = list()
    type_v = list()
    type_idx = list()
    start_frame = list()
    end_frame = list()
    frame_num = list()
    length = list()
    mi = 0
    ma = 0
    sss = list()
    before = list()
    after = list()
    for i in range(len(df_sheet['Subject'].values)):
        videoname = df_sheet['Filename'].values[i]
        startframe = df_sheet['Onset'].values[i]
        endframe = df_sheet['Offset'].values[i]
        label = df_sheet['Expression type'].values[i]

        # find full path of the video about original videos
        videoname = videoname.lower() if videoname.isupper() else videoname 
        ex = os.path.join(df_sheet['Subject'].values[i], videoname, 'color')
        path_full = os.path.join(v_path, ex)

        # according to gt, devide videos into two parts: labeled videos ang unlabeled videos
        all_videoname = set(os.listdir(os.path.join(v_path, df_sheet['Subject'].values[i])))
        labeled_videoname = set((df_sheet[df_sheet['Subject'].values == df_sheet['Subject'].values[i]])['Filename'].values)
        unlabeled_videoname = list(all_videoname - labeled_videoname)
    
        # the savepath of processed data
        if extract_full_data:
            basepath = os.path.join(path_full.split('/part_A')[0], 'new_data_full_reduce_more')
        else:
            basepath = os.path.join(path_full.split('/part_A')[0], 'new_data_train')

        new_subject = df_sheet['Subject'].values[i].split('.')[-1].zfill(3)
        video_save_path = os.path.join(basepath, new_subject, videoname)
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        
        # processing one training video 
        sss, before, after, firstname = process_one_video(path_full, video_save_path, sss, before, after, startframe, endframe)
        # including all videos into new data folder
        if extract_full_data: 
            for vn in unlabeled_videoname:
                vsp_all = os.path.join(basepath, new_subject, vn)
                if not os.path.exists(vsp_all):
                    os.makedirs(vsp_all)
                vn_all = os.path.join(v_path, df_sheet['Subject'].values[i], vn, 'color')
                if len(os.listdir(os.path.join(basepath, new_subject, unlabeled_videoname[0]))) < len(os.listdir(vn_all)):
                    sss, before, after, firstname = process_one_video(vn_all, vsp_all, sss, before, after)

        count_pic = len(os.listdir(video_save_path))
        real_video_path = video_save_path
        real_sub =  new_subject
        real_start = int(startframe) - firstname + 1
        real_end = int(endframe) - firstname + 1
        # accoding to the paper： Emotions revealed: recognizing faces and feelings to improve communication and emotional life
        if real_end <= count_pic and delete_long:
            if (label == 'Micro-expression' and real_end - real_start <= 15) or (label == 'Macro-expression' and real_end - real_start <= 120):
                subject.append(real_sub)
                video_names.append(real_video_path)
                start_frame.append(real_start)
                end_frame.append(real_end)
                frame_num.append(int(count_pic))
                length.append(int(real_end - real_start))
                if label =='Macro-expression':
                    t_idx = 1
                    ma = ma + 1
                    label = 'Macro'
                elif label == 'Micro-expression':
                    t_idx = 2
                    label = 'Micro'
                    mi = mi +1
                type_v.append(label)
                type_idx.append(t_idx)

        elif real_end <= count_pic and not delete_long:
            subject.append(real_sub)
            video_names.append(real_video_path)
            start_frame.append(real_start)
            end_frame.append(real_end)
            frame_num.append(int(count_pic))
            length.append(int(real_end - real_start))
            if label =='Macro-expression':
                t_idx = 1
                ma = ma + 1
                label = 'Macro'
            elif label == 'Micro-expression':
                t_idx = 2
                label = 'Micro'
                mi = mi +1
            type_v.append(label)
            type_idx.append(t_idx)
        else:
            print('Out of range', path_full, startframe)
    
    dic_oom = dict()
    dic_oom['path'] = sss
    dic_oom['before'] = before
    dic_oom['after'] = after
    df_oom = pd.DataFrame(dic_oom, columns=['path', 'before', 'after'])
    df_oom.sort_values(['path', 'before'], inplace=True)
    if extract_full_data and delete_long:
        df_oom.to_csv('cas3_oom_full_reduce.csv', encoding='utf-8', index=False)
    elif extract_full_data and not delete_long:
        df_oom.to_csv('cas3_oom_full.csv', encoding='utf-8', index=False)
    else:
        df_oom.to_csv('cas3_oom_training.csv', encoding='utf-8', index=False)
    
    dic_inf = dict()
    dic_inf['subject'] = subject
    dic_inf['video'] = video_names
    dic_inf['type'] = type_v
    dic_inf['type_idx'] = type_idx
    dic_inf['startFrame'] = start_frame
    dic_inf['endFrame'] = end_frame
    dic_inf['frame_num'] = frame_num
    dic_inf['length'] = length

    df = pd.DataFrame(dic_inf, columns=['subject', 'video', 'type', 'type_idx', 'startFrame', 'endFrame', 'frame_num', 'length'])
    df.sort_values(['video', 'type_idx', 'startFrame'], inplace=True)

    if extract_full_data and delete_long:
        df_oom.to_csv('cas3_annotation_full_reduce.csv', encoding='utf-8', index=False)
    elif extract_full_data and not delete_long:
        df_oom.to_csv('cas3_annotation_full.csv', encoding='utf-8', index=False)
    else:
        df_oom.to_csv('cas3_annotation_full.csv', encoding='utf-8', index=False)
    print('number of mi', mi, 'number of ma', ma)
    
    count_len(length, l_path)

def count_len(lenth, length_path):
    length_path_one = length_path.replace('len', 'one', 1)
    with open(length_path_one, 'a') as f:
        for i in lenth:
            f.write(str(i))
            f.write('\n')
    lenth.sort()

    dicts = {x: lenth.count(x) for x in set(lenth)}
    with open(length_path, 'a') as f:
        for k, v in dicts.items():
            f.write(str(k))
            f.write(' ')
            f.write(str(v))
            f.write('\n')    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-excel_path', type=str, default='/home/yww/CAS^3/part_A/annotation/CAS(ME)3_part_A_v2.xls')
    parser.add_argument('-video_files', type=str, default='/home/yww/CAS^3/part_A/data/part_A')
    parser.add_argument('-len_path', type=str, default='./cas(me)3_len_reduce_more.txt')
    parser.add_argument('-delete_long_microexpression', action='store_false', default=True, help='if true, delete long micro-expressions which are long than 0.5s')
    parser.add_argument('-extract_full_data', action='store_false', default=True, help='if true, extract all videos, or extract labeled videos')
    args = parser.parse_args()

    excel_path = args.excel_path
    video_files = args.video_files
    len_path = args.len_path
    extract_full_data = args.extract_full_data
    delete_lme = args.delete_long_microexpression

    main(excel_path, video_files, len_path, extract_full_data, delete_lme)
