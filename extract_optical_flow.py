import os
import cv2
print('OpenCL available:', cv2.ocl.haveOpenCL())
import glob
import numpy as np
import argparse
from tqdm import tqdm
import time


def cal_for_frames(v_path, dataset, f_path=''):
    frames = glob.glob(os.path.join(v_path, '*.jpg')) + glob.glob(os.path.join(v_path, '*.png'))
    if dataset=='cas(me)^2':
        frames = [int(str(i).split('_')[-1].split('.')[0]) for i in frames]
        frames.sort()
        frames = [v_path + '/img_' + str(i) + '.jpg' for i in frames]
    elif dataset=='cas(me)^3':
        frames = [int(str(i).split('/'+ v_path[-1]+ '/')[-1].split('.')[0]) for i in frames]
        frames.sort()
        frames = [v_path + '/' + str(i) + '.jpg' for i in frames]
    else:
        frames.sort()
    flow = []
    dif_list = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    # prev = cv2.equalizeHist(prev)

    # prev = cv2.medianBlur(prev, 5)
    # prev = cv2.adaptiveThreshold(prev, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # prev = cv2.adaptiveThreshold(prev, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    for frame_curr in tqdm(frames[1:], desc=v_path):
        
        prev_h, prev_w = prev.shape[0], prev.shape[1]
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_resize = cv2.resize(curr, (prev_w, prev_h))
        # curr = cv2.equalizeHist(curr)

        dif = np.abs(cv2.subtract(curr_resize, prev))
        dif = np.power(dif, 2)
        dif_sum = np.sum(dif)
        dif_list.append(dif_sum)
        # curr = cv2.medianBlur(curr, 5)
        # curr_b = cv2.adaptiveThreshold(curr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # curr = cv2.add(curr, curr_b)
        # curr = cv2.adaptiveThreshold(curr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        tmp_flow = compute_TVL1(prev, curr_resize)
        flow.append(tmp_flow)
        prev = curr
        frame_index = frames[1:].index(frame_curr)
        if dataset == 'samm' and (frame_index%100==99 or frame_index==(len(frames[1:])-1)):
            save_flow_samm(flow, f_path, dif_list, frame_index-frame_index%100)
            flow = list()
            dif_list = list()
            
    return flow, dif_list


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""

    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1=cv2.createOptFlow_DualTVL1()
    prev = cv2.UMat(prev)
    curr = cv2.UMat(curr)
    flow = TVL1.calc(prev, curr, None)
    flow = cv2.UMat.get(flow)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path, dif):
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.makedirs(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.makedirs(os.path.join(flow_path, 'v'))
    dif_f = open(os.path.join(flow_path, 'dif.txt'), 'a')
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, 'u', "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, 'v', "{:06d}.jpg".format(i)),
                    flow[:, :, 1])
    for i, dif in enumerate(dif):
        dif_f.write(str(i)+':'+str(dif))
        dif_f.write('\n')
    dif_f.close()


def save_flow_samm(video_flows, flow_path, dif, num):
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.makedirs(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.makedirs(os.path.join(flow_path, 'v'))
    dif_f = open(os.path.join(flow_path, 'dif.txt'), 'a')
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, 'u', "{:06d}.jpg".format(i+num)),flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, 'v', "{:06d}.jpg".format(i+num)),flow[:, :, 1])
    for i, dif in enumerate(dif):
        dif_f.write(str(i)+':'+str(dif))
        dif_f.write('\n')
    dif_f.close()



def extract_flow(vi_path, flow_path, dataset):
    video_sub = os.listdir(vi_path)
    if dataset=='cas(me)^2' or dataset=='cas(me)^3':
        for i in video_sub:
            one_subject = os.path.join(vi_path, i)
            for j in (os.listdir(one_subject)):
                a = glob.glob(os.path.join(os.path.join(vi_path, i, j), '*.jpg')) + glob.glob(os.path.join(os.path.join(vi_path, i, j), '*.png'))
                b = glob.glob(os.path.join(os.path.join(os.path.dirname(vi_path), os.path.basename(flow_path), i, j, 'u'), '*.jpg')) 
                if len(a)==0:
                    print(os.path.join(vi_path, i, j))
                if len(a) != len(b)+1:
                    flow, dif = cal_for_frames(os.path.join(vi_path, i, j), dataset)
                    f_path = os.path.join(flow_path, i, j)
                    save_flow(flow, f_path, dif)
                    print('complete:' + f_path)
                print(os.path.join(os.path.dirname(vi_path), os.path.basename(flow_path), i, j, 'u'))
    else:
        for i in video_sub:
            f_path = os.path.join(flow_path, i)
            flow, dif = cal_for_frames(os.path.join(vi_path, i), dataset, f_path)
            print('complete:' + f_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--amply', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='cas(me)^3')
    parser.add_argument('--video_path', type=str, default='/home/yww/CAS^3/new_data_full_reduce')
    parser.add_argument('--save_path', type=str, default='/home/yww/CAS^3/new_data_flow')
    args = parser.parse_args()
    
    video_path = args.video_path
    save_path = args.save_path
    dataset = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    extract_flow(video_path, save_path, dataset)
