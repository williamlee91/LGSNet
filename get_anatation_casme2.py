import pandas as pd
import os
import argparse


def main(e_path, v_path, l_path):
    df_sheet1 = pd.read_excel(e_path, sheet_name=0, header=None)
    df_sheet2 = pd.read_excel(e_path, sheet_name=1, header=None)
    df_sheet3 = pd.read_excel(e_path, sheet_name=2, header=None)

    subject = list()
    video_names = list()
    type_v = list()
    type_idx = list()
    start_frame = list()
    end_frame = list()
    frame_num = list()
    length = list()
    for i in range(len(df_sheet1[1].values)):
        expression = df_sheet1[1].values[i]
        startframe = df_sheet1[2].values[i]
        endframe = df_sheet1[4].values[i]
        label = df_sheet1[7].values[i]
        if int(endframe) == 0 or endframe<=df_sheet1[3].values[i]:
            if int(endframe) != 0:
                print('EOF', df_sheet1[7].values[i], startframe)
            endframe = df_sheet1[3].values[i] + 20
        if label =='macro-expression':
            t_idx = 1
        elif label == 'micro-expression':
            t_idx = 2
        assert(int(df_sheet1[0].values[i]) == int(df_sheet2[2].values[i]))
        file_index = df_sheet2[1].values[i]

        # find full path of the video about ex
        ex = str(expression)
        # this method is wrong!!!
        # if ex[:-2] in df_sheet3[1].values:
        #     num_index = list(df_sheet3[1].values).index(ex[:-2])
        # this method is right!!!
        if ex.split('_')[0] in df_sheet3[1].values:
            num_index = list(df_sheet3[1].values).index(ex.split('_')[0])
            subname = '0' + str(list(df_sheet3[0].values)[num_index])
            # the first half of video path
            subname_full = str(file_index) + '_' + subname
            # determine whether the first half path is in the directory
            file_path = os.path.join(v_path, file_index)
            video_name = os.listdir(file_path)
            video_name_part = [name[:7] for name in video_name]
            # delete 's'
            if subname_full[1:] in video_name_part:
                file_name = video_name[video_name_part.index(subname_full[1:])]
                # full video path
                path_full = os.path.join(v_path, file_index, file_name)
                all_pic = os.listdir(path_full)
                # sort pic paths
                all_pic = [int(str(i).split('_')[1].split('.')[0]) for i in all_pic]
                all_pic.sort()
                all_pic = ['/img_' + str(i) + '.jpg' for i in all_pic]

                count_pic = len(all_pic)

                subject.append(df_sheet1[0].values[i])
                video_names.append(path_full)
                type_v.append(label)
                type_idx.append(t_idx)
                start_frame.append(int(startframe))
                end_frame.append(int(endframe))

                # frames_path = os.path.join(vid_data_dir, vid_name)
                # frames_set = os.listdir(frames_path)
                # assert len(frames_set) % 3 == 0
                # frame_num.append(int(len(frames_set) / 3))
                frame_num.append(int(count_pic))
                length.append(int(endframe-startframe))
                print(i)
        else:
            print(i)
            print('11111111111111111', ex[:-2])

    dic_inf = dict()
    dic_inf['subject'] = subject
    dic_inf['video'] = video_names
    dic_inf['type'] = type_v
    dic_inf['type_idx'] = type_idx
    dic_inf['startFrame'] = start_frame
    dic_inf['endFrame'] = end_frame
    dic_inf['frame_num'] = frame_num
    dic_inf['length'] = length
    count_len(length, l_path)

    df = pd.DataFrame(dic_inf, columns=['subject', 'video', 'type', 'type_idx', 'startFrame',  'endFrame', 'frame_num', 'length'])
    df.sort_values(['video', 'type_idx', 'startFrame'], inplace=True)
    df.to_csv('./casme2_annotation.csv', encoding='utf-8', index=False)

    
def count_len(lenth, length_path):
    with open('./cas(me)2_len_all_1111.txt', 'a') as f:
        for i in lenth:
            f.write(str(i))
            f.write('\n')
    dicts = {x: lenth.count(x) for x in set(lenth)}
    with open(length_path, 'a') as f:
        for k, v in dicts.items():
            f.write(str(k))
            f.write(' ')
            f.write(str(v))
            f.write('\n')
    
    large = 0
    middle = 0
    small = 0

    for i in lenth:
        if i <=40:
            small = small + 1
        elif 40<i<=80:
            middle = middle + 1
        else:
            large = large + 1
    
    print(large, middle, small)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-excel_path', type=str, default='./CAS(ME)^2code_final.xlsx')
    parser.add_argument('-video_files', type=str, default='./CAS(ME)2_longVideoFaceCropped/longVideoFaceCropped')
    parser.add_argument('-len_path', type=str, default='./cas(me)2_len.txt')
    args = parser.parse_args()
        
    args = parser.parse_args()

    excel_path = args.excel_path
    video_files = args.video_files
    len_path = args.len_path

    main(excel_path, video_files, len_path)
