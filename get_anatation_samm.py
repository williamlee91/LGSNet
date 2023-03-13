import pandas as pd
import os
import argparse
import math

def main(e_path, v_path, l_path):
    
    df_sheet = pd.read_excel(e_path, header=9)

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
    for i in range(len(df_sheet['Subject'].values)):

        expression = df_sheet['Filename'].values[i]
        startframe = df_sheet['Onset'].values[i]
        endframe = df_sheet['Offset'].values[i]
        label = df_sheet['Type'].values[i]
        if expression in ['012_4_1', '020_6_5','036_7_4']:
            continue

        # find full path of the video about ex
        ex = str(expression[:5])
        
        path_full = os.path.join(v_path, ex)
        all_pic = os.listdir(path_full)
        count_pic = len(all_pic)
               
        all_pic = [int(str(i).split('_')[2].split('.')[0]) for i in all_pic]
        all_pic.sort()

        if int(endframe) <= all_pic[-1] and (endframe-startframe)<=800:
            subject.append(expression[:3])
            video_names.append(path_full)
            start_frame.append(int(startframe))
            end_frame.append(int(endframe))
            frame_num.append(int(count_pic))
            length.append(int(endframe-startframe))
            if label =='Macro':
                t_idx = 1
                ma = ma + 1
            elif label == 'Micro - 1/2':
                t_idx = 2
                label = 'Micro'
                mi = mi +1
            type_v.append(label)
            type_idx.append(t_idx)
        else:
            print(expression)
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
    df.to_csv('samm_annotation_22.csv', encoding='utf-8', index=False)
    print('number of mi', mi, 'number of ma', ma)
    count_len(length, l_path)

def count_len(lenth, length_path):
    with open('./samm_len_all_22_22.txt', 'a') as f:
        for i in lenth:
            f.write(str(i))
            f.write('\n')
    lenth.sort()
    lenth = [math.ceil(i*3/20) for i in lenth]
    dicts = {x: lenth.count(x) for x in set(lenth)}
    with open(length_path, 'a') as f:
        for k, v in dicts.items():
            if k<512:
                f.write(str(k))
                f.write(' ')
                f.write(str(v))
                f.write('\n')    
    middle = 0
    small = 0
    for i in lenth:
        if i <=101:
            small = small + 1
        elif 101<i:
            middle = middle + 1
    
    print(middle, small)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-excel_path', type=str, default='./SAMM_LongVideos_V2_Release.xlsx')
    parser.add_argument('-video_files', type=str, default='./SAMM_longvideos')
    parser.add_argument('-len_path', type=str, default='./samm_len_22_22.txt')
    args = parser.parse_args()
        
    args = parser.parse_args()

    excel_path = args.excel_path
    video_files = args.video_files
    len_path = args.len_path

    main(excel_path, video_files, len_path)
