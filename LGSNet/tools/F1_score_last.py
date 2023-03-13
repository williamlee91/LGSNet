import os
import glob
import pandas as pd
import numpy as np
import argparse

def all_score(TP1,TP2,N1,N2,recall_all):

    if (N1 ==0 and N2==0) or (TP1==0 and TP2 ==0):
        precision_all =0
        F1_SCORE = 0
    else:
        precision_all = 1.0* (TP1+TP2)/(N1+N2)
        F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)
    return F1_SCORE,precision_all
   
def main_threshold(path, dataset, annotation, version, label_frequency, start_threshold, max_num_pos, base_duration):
    print(path)
    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key = lambda x:int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()

    best_ME_1, best_ME_2 = 0, 0
    best, best_m1, best_m2 = 0.10, 0.15, 0.01
    best_recall = 0
    if dataset =='cas(me)^2':
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'threshold', 'cathreshold'+'_'+str(version))
    elif dataset =='cas(me)^3':
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'threshold', 'ca3threshold'+'_'+str(version))
    else:
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'threshold', 'sathreshold'+'_'+str(version))
    if not os.path.exists(out_path_tmp):
        os.makedirs(out_path_tmp)
    best_out = os.path.join(out_path_tmp, os.path.basename(path)+'_best_sample.log')
    threshold_out = os.path.join(out_path_tmp, os.path.basename(path)+'_'+path[-2:]+'_threshlod.log')
    if os.path.exists(threshold_out):
        os.remove(threshold_out)
    tp_sub = dict()
    for e in range(5, 120):
        txt_index = txts[e]
        # all subjects in the same epoch
        test_path = [os.path.join(i, 'test_'+str(txt_index).zfill(2)+'.txt') for i in test_path_temp]   
        # confirm the best threshold
        for k_temp in range(start_threshold, 500, 1):
            k = 1.0 *k_temp/1000  
            TP1, TP2 = 0, 0
            N1, N2, N_all, N2_ME = 0, 0, 0, 0
            length_count = list()
            write_list = list()
            length_pre = list()
            # every subject in one file (200x)
            for ip in test_path:
                T = 0
                with open(ip, 'r') as f:
                    all_lines = f.readlines()
                if not all_lines:
                    continue
                all_lines = [h.split('\t') for h in all_lines]
                # divide all gts of every video
                tmp_video = all_lines[0][0]
                count = 1
                tmp_list = list()
                all_test = dict()
                all_video = list(set([name[0] for name in all_lines]))
                for tv in all_video:
                    tmp_video = tv
                    for j in range(len(all_lines)):
                        if all_lines[j][0] == tmp_video:
                            tmp_list.append(all_lines[j])
                    all_test[count] = tmp_list
                    count = count + 1
                    tmp_list = list()
                # number of GT of every video
                num_of_video = len(all_test.keys()) 
                # least len of GT
                part_tmp = list()
                # select predictions of every video (prob > threshold)
                for i in range(num_of_video):
                    tmp_one_video = list(all_test.values())[i]
                    part = [o for o in tmp_one_video if float(o[-1][:-2]) > k ]
                    # N1: number of precictions of macro-expressions
                    # N2: number of precictions of micro-expressions
                    # N_all: number of precictions
                    if len(part) > max_num_pos :
                        part = part[:max_num_pos]
                    N_all = N_all + len(part)
                    N1 = N1 + len([o for o in part if np.array(o[2]).astype(float).astype(np.int64) * int(label_frequency)-np.array(o[1]).astype(float).astype(np.int64) * int(label_frequency) > base_duration])
                    N2 = N2 + len([o for o in part if np.array(o[2]).astype(float).astype(np.int64) * int(label_frequency)-np.array(o[1]).astype(float).astype(np.int64) * int(label_frequency) <= base_duration])
                    N2_ME = N2_ME + len([o for o in part if np.array(o[2]).astype(float).astype(np.int64) * int(label_frequency)-np.array(o[1]).astype(float).astype(np.int64) * int(label_frequency) <= 2 *base_duration])
                    if not part:
                        part = [[tmp_one_video[0][0], '100000', '100000', '_','_']]
                    part_tmp.append(part)   
                part_pre = part_tmp

                # predictions: sorted by prob
                part_pre= [sorted(i, key = lambda x:int(float(x[1]))) for i in part_pre]
                
                # calculate iou between every prediction with GT
                for video_num, pre in enumerate(part_pre):
                    video_name_list = list(set(ann_csv.video.values[:].tolist()))
                    video_name_list.sort()
                    
                    # identify the current video
                    video_name_last = part_pre[video_num][0][0]
                    if dataset =='cas(me)^2':
                        video_name_part = 's' + video_name_last[:2]
                        video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)
                    elif dataset == 'samm_merge':
                        video_name = os.path.join(video_name_list[0][:-4],str(video_name_last).zfill(3))
                    elif dataset == 'cas(me)^3':
                        video_name = os.path.join(video_name_list[0][:-6], str(video_name_last.split('_')[0]).zfill(3), str(video_name_last.split('_')[1]))
                    else:
                        video_name = os.path.join(video_name_list[0][:-6],str(video_name_last).zfill(3))
                    # select startframes of current video
                    video_ann_df = ann_csv[ann_csv.video == video_name]
                    act_start_video = video_ann_df['startFrame'].values[:]
                    # select indexes of startframes of current video
                    indexes = np.argsort(act_start_video)
                    # labels and endframes are sorted by indexes from actual start frames
                    act_end_video = video_ann_df['endFrame'].values[:]
                    act_end_video = np.array(act_end_video)[indexes]
                    # labels = video_ann_df['type_idx'].values[:]
                    # labels = np.array(labels)[indexes]
                    # actual start frames are sorted by time series
                    act_start_video.sort()
                    
                    pre = np.array(pre)
                    pre_start = pre[:,1].astype(float).astype(np.int64) * int(label_frequency)
                    pre_end = pre[:,2].astype(float).astype(np.int64) * int(label_frequency)
                    
                    start_tmp = list()
                    end_tmp = list()
                    for m in range(len(act_start_video)):
                        video_label = video_name_last[:7]
                        act_start = int(act_start_video[m])
                        act_end = int(act_end_video[m])
                        iou = (np.minimum(pre_end, act_end) - np.maximum(pre_start, act_start)+1)/(np.maximum(pre_end, act_end) - np.minimum(pre_start, act_start)+1)
                        max_iou = np.max(iou)
                        max_index = np.argmax(iou)
                        if max_iou >= 0.5:
                            tmp_write_list = [video_label, pre_start[max_index], pre_end[max_index], act_start, act_end, 'TP']
                            write_list.append(tmp_write_list)  
                            length_count.append(act_end-act_start)
                            length_pre.append(pre_end[max_index]-pre_start[max_index])
                            if act_end - act_start > base_duration:
                                TP1 = TP1 + 1
                            elif 0 < act_end - act_start <= base_duration:
                                TP2 = TP2 + 1
                            start_tmp.append(pre_start[max_index])
                            end_tmp.append(pre_end[max_index])
                            T = T + 1
                        else:
                            tmp_write_list = [video_label, '_', '_', act_start, act_end, 'FP']
                            write_list.append(tmp_write_list) 
                    pre_start_remain = list(pre_start)
                    pre_end_remain = list(pre_end)
                    pre_remain_s = [i for i in pre_start_remain if i not in start_tmp] 
                    pre_remain_e = [i for i in pre_end_remain if i not in end_tmp] 
                    try:
                        if pre_remain_s[0] < 100000 and len(pre_remain_s) == len(pre_remain_e):
                            write_remain = [[video_label, i, pre_end[pre_start==i][0], '_', '_', 'FN'] for i in pre_remain_s]
                            write_list = write_list + write_remain
                        elif pre_remain_s[0] == 100000:
                            pass
                        else:
                            write_remain = [[video_label, pre_start[pre_end==i][0], i, '_', '_', 'FN'] for i in pre_remain_e]
                            write_list = write_list + write_remain
                    except:
                        pass
                num_index = ip.find('subject') 
                sub_num = ip[num_index+9:num_index+11]
                tp_key = str(e)+'_'+ sub_num
                try:
                    tp_sub[tp_key] 
                except:
                    tp_sub[tp_key] = 0     
                if T > tp_sub[tp_key]:
                    tp_sub[tp_key] =  T    
            # calculate F1_score
            # M_all need to calculate in SAMM
            # M1： Number of macro-expressions
            # M2： Number of micro-expressions
            if dataset == 'cas(me)^2' or dataset == 'cas(me)^2_merge':
                M1 = 300
                M2 = 57
            elif dataset == 'cas(me)^3':
                M1 = 2071
                M2 = 277
            else:
                # M1 = 340
                # M2 = 159
                M1 = 312
                M2 = 159
            recall1 = 1.0* TP1/M1
            recall2 = 1.0* TP2/M2
            recall_all = 1.0 *(TP1+TP2)/(M1+M2)
            if recall_all > best_recall:
                best_recall = recall_all
                print('best', recall_all)
            # Sometimes, there are no predictions of micro-expressions or macro-expressions
            F1_SCORE, precision_all = all_score(TP1,TP2,N1,N2,recall_all)
            if recall1 > best_m1:
                best_m1 = recall1
                print("recall_macro: %05f, recall_micro: %05f, %f"%(best_m1, best_m2, k))
            if recall2 > best_m2:
                best_m2 = recall2
                print("recall_macro: %05f, recall_micro: %05f, %f"%(best_m1, best_m2, k))
            # record best the F1_scroe and the result of predictions
            if F1_SCORE > best:
                best = F1_SCORE
                # print('number of epoch: %d, threshold: %5f'%(e, k))
                print("recall: %05f, precision: %05f, f1_score: %05f"%(recall_all, precision_all, best))
                with open(best_out, 'w') as f_sout:
                    f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                with open(threshold_out, 'a') as f_threshold:
                    f_threshold.writelines("%d, %f, %d, %d, %d, %d, f1: %05f, recall: %05f, precision: %05f, f1_score_macro: %05f, f1_score_micro: %05f\n" 
                                % (e, k, TP1, TP2, N1, N2, F1_SCORE, recall_all, precision_all, best_m1, best_m2))
                if best > 0.38:
                    standard_out = os.path.join(out_path_tmp, os.path.basename(path)+'_'+str(e)+'_'+str(k)+'_'+str(F1_SCORE)+'.log')
                    with open(standard_out, 'w') as f_sout:
                        f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                length_count.sort()
                length_pre.sort()
                print('pre:', length_pre,'\n','act:', length_count,'\n',TP1, TP2, N1, N2, k) 
            
            F1_SCORE_ME_1 = 2.0*(TP2/N2 * TP2/M2)/(TP2/N2 + TP2/M2) if TP2!=0 and N2!=0 else 0
            F1_SCORE_ME_2 = 2.0*(TP2/N2_ME * TP2/M2)/(TP2/N2_ME + TP2/M2) if TP2!=0 and N2_ME!=0 else 0
            if F1_SCORE_ME_1 > best_ME_1:
                print('ME1',F1_SCORE_ME_1)
                best_ME_1 = F1_SCORE_ME_1
            if F1_SCORE_ME_2 > best_ME_2:
                print('ME2',F1_SCORE_ME_2)
                best_ME_2 = F1_SCORE_ME_2
        print("epoch:  !!!!!!!!!!!!!!!!!!!!!!!!", e+1)    
    print(tp_sub)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')

    # parser.add_argument('--path', type=str, default='/home/yww/LGSNET/LGSNet/output/cas(me)^2_new')
    # parser.add_argument('--ann', type=str, default='/home/yww/LGSNET/LGSNet/casme2_annotation.csv')
    # parser.add_argument('--dataset', type=str, default='cas(me)^2')
    # # parser.add_argument('--ann', type=str, default=r'/home/yww/LGSNET/LGSNet/casme2_annotation_357.csv')
    # # parser.add_argument('--dataset', type=str, default=r'cas(me)^2_merge')
    # parser.add_argument('--version', type=int, default=28)
    # parser.add_argument('--top_k', type=bool, default=False)
    # parser.add_argument('--label_frequency', type=float, default=1.0)
    # parser.add_argument('--start_threshold', type=int, default=300)
    # parser.add_argument('--most_pos_num', type=int, default=14)
    # parser.add_argument('--base_duration', type=int, default=15)

    # parser.add_argument('--path', type=str, default='/home/yww/LGSNET/LGSNet/output/samm')
    # parser.add_argument('--ann', type=str, default='/home/yww/LGSNET/LGSNet/samm_annotation_merge_part_2000_L800_new2.csv')
    # parser.add_argument('--dataset', type=str, default='samm')
    # parser.add_argument('--version', type=int, default=28)
    # parser.add_argument('--top_k', type=bool, default=False)
    # parser.add_argument('--label_frequency', type=float, default=1.0)
    # parser.add_argument('--start_threshold', type=int, default=100)
    # parser.add_argument('--most_pos_num', type=int, default=14)
    # parser.add_argument('--base_duration', type=int, default=15)

    parser.add_argument('--path', type=str, default='/home/yww/LGSNET/LGSNet/output/cas(me)^3')
    parser.add_argument('--ann', type=str, default='/home/yww/LGSNET/LGSNet/cas3_annotation_full_me_reduce.csv')
    parser.add_argument('--dataset', type=str, default='cas(me)^3')
    parser.add_argument('--version', type=int, default=28)
    parser.add_argument('--top_k', type=bool, default=False)
    parser.add_argument('--label_frequency', type=float, default=1.0)
    parser.add_argument('--start_threshold', type=int, default=300)
    parser.add_argument('--most_pos_num', type=int, default=14)
    parser.add_argument('--base_duration', type=int, default=15)

    args = parser.parse_args()
    path = args.path
    dataset = args.dataset
    ann = args.ann
    version = args.version
    top_k = args.top_k
    label_frequency = args.label_frequency
    start_threshold = args.start_threshold
    max_num_pos = args.most_pos_num
    base_duration = args.base_duration
 
    main_threshold(path, dataset, ann, version, label_frequency, start_threshold, max_num_pos, base_duration)