import os
from pickle import TRUE
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import io
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse
import torchvision
from PIL import Image
import numpy as np
from pytorch_i3d import InceptionI3d
import cv2
import pdb


def load_frame(frame_file, dataset, resize=False):

    data = Image.open(frame_file)

    if dataset=='cas(me)^2':
        assert(data.size[1] == 400)
        assert(data.size[0] == 400)
    elif dataset=='samm':
        # warming: there are some pics with 432x540 in the video of 032
        try:
            assert(data.size[1] == 600)
            assert(data.size[0] == 600)
        except:
            print(frame_file, data.size[0], data.size[1])
        
    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max() <= 1.0)
    assert(data.min() >= -1.0)

    return data

def oversample_data(data):  # (39, 16, 224, 224, 2)  # Check twice

    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])   # ,:,16:240,58:282,:
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


def load_rgb_batch(frames_dir, rgb_files, frame_indices, dataset, resize=False):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            try:
                batch_data[i, j, :, :, :] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]), dataset, resize)
            except:
                batch_data[i, j, :, :, 0] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]), dataset, resize)
                batch_data[i, j, :, :, 1] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]), dataset, resize)
                batch_data[i, j, :, :, 2] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]), dataset, resize)

    return batch_data

def load_flow_batch(frames_dir, flow_x_files, flow_y_files, frame_indices, dataset, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            # batch_data[i, j, :, :, 0] = load_frame(os.path.join(frames_dir, flow_x_files[frame_indices[i][j]]), resize)
            # batch_data[i, j, :, :, 1] = load_frame(os.path.join(frames_dir, flow_y_files[frame_indices[i][j]]), resize)
            batch_data[i, j, :, :, 0] = load_frame(os.path.join(os.path.join(frames_dir, 'u'), flow_x_files[frame_indices[i][j]]), dataset, resize)
            batch_data[i, j, :, :, 1] = load_frame(os.path.join(os.path.join(frames_dir, 'v'), flow_y_files[frame_indices[i][j]]), dataset, resize)
    return batch_data

def load_zipframe(zipdata, name, resize=False):

    stream = zipdata.read(name)
    data = Image.open(io.BytesIO(stream))

    assert(data.size[1] == 400)
    assert(data.size[0] == 400)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max() <= 1.0)
    assert(data.min() >= -1.0)

    return data

def load_ziprgb_batch(rgb_zipdata, rgb_files, frame_indices, resize=False):

    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_zipframe(rgb_zipdata, rgb_files[frame_indices[i][j]], resize)
    return batch_data

def load_zipflow_batch(flow_x_zipdata, flow_y_zipdata, flow_x_files, flow_y_files, frame_indices, resize=False):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224, 224, 2))
    else:
        batch_data = np.zeros(frame_indices.shape + (256, 340, 2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, 0] = load_zipframe(flow_x_zipdata, flow_x_files[frame_indices[i][j]], resize)
            batch_data[i, j, :, :, 1] = load_zipframe(flow_y_zipdata, flow_y_files[frame_indices[i][j]], resize)
    return batch_data


def run(dataset='cas', mode='rgb', load_model='', sample_mode='oversample', frequency=8,
        input_dir='', output_dir='', batch_size=40, usezip=False, chunk_size=32, merge=True):

    assert(mode in ['rgb', 'flow'])
    assert(sample_mode in ['oversample', 'center_crop', 'resize'])
    
    # setup the model
    if mode == 'rgb':
        i3d = InceptionI3d(400, in_channels=3)
    else:
        i3d = InceptionI3d(400, in_channels=2)

    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224

        with torch.no_grad():
            b_data = Variable(b_data.cuda()).float()
        # b_data = Variable(b_data.cuda(), volatile=True).float()
        b_features = i3d.extract_features(b_data)
        
        b_features = b_features.data.cpu().numpy()[:, :, 0, 0, 0]
        return b_features

    video_list = list()
    # the paths of all videos
    if dataset == 'cas' and not merge:
        subject_names_list = os.listdir(input_dir)
        for i in subject_names_list:
            video_name_list = os.listdir(os.path.join(input_dir, i))
            video_names_sub = [os.path.join(input_dir,i, j) for j in video_name_list]
            if mode =='rgb':
                video_list = video_names_sub + video_list
            else:   
                video_names = [k for k in video_names_sub if 'v' in os.listdir(k)]
                video_list = video_names_sub + video_list
    
    else:
        subject_names_list = os.listdir(input_dir)
        if mode == 'rgb':
            video_list = [os.path.join(input_dir, i) for i in subject_names_list]
        else:
            video_names_sub = [os.path.join(input_dir, j) for j in subject_names_list]
            video_names = [k for k in video_names_sub if 'v' in os.listdir(k)]
            video_list = video_names_sub + video_list
    
    video_names = video_list
    for video_name in video_names:
        # cas(me)^2, it should creat sub_file: s15, s16, s19.... 
        if dataset =='cas' and not merge:
            subject = video_name.split('/')[-2]
            v_name = video_name.split('/')[-1]
            subject_path = os.path.join(output_dir, subject)
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)
        # samm, it will be saved directly
        else:
            v_name = video_name.split('/')[-1]
            subject_path = output_dir
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)
        # the name of the feature to be saved
        save_file = '{}-{}.npz'.format(v_name, mode)
        if save_file in os.listdir(subject_path):
            continue

        # frames_dir = os.path.join(input_dir, video_name)
        frames_dir = video_name
        # all pic paths in one video
        if mode == 'rgb':
            if usezip:
                rgb_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'img.zip'), 'r')
                rgb_files = [i for i in rgb_zipdata.namelist() if i.startswith('img')]
            else:
                rgb_files = [i for i in os.listdir(frames_dir) if i.endswith('.jpg')]
                
            # All pics should be sorted
            if dataset == 'cas' :
                rgb_files = [int(str(i).split('_')[-1].split('.')[0]) for i in rgb_files] 
                rgb_files.sort()
                rgb_files = ['img_' + str(i) + '.jpg' for i in rgb_files[:-1]] if os.listdir(frames_dir)[0].startswith('img_') else [str(i) + '.jpg' for i in rgb_files[:-1]] 
            else:
                rgb_files.sort() 
                rgb_files = rgb_files[:-1]                  
            frame_cnt = len(rgb_files)
        # all optical flow paths in one video
        else:
            if usezip:
                flow_x_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_x.zip'), 'r')
                flow_x_files = [i for i in flow_x_zipdata.namelist() if i.startswith('x_')]

                flow_y_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_y.zip'), 'r')
                flow_y_files = [i for i in flow_y_zipdata.namelist() if i.startswith('y_')]
            else:
                flow_x_files = [i for i in os.listdir(os.path.join(frames_dir, 'u')) if i.startswith('0')]
                flow_y_files = [i for i in os.listdir(os.path.join(frames_dir, 'v')) if i.startswith('0')]

            flow_x_files.sort()
            flow_y_files.sort()
            assert(len(flow_y_files) == len(flow_x_files))
            frame_cnt = len(flow_y_files)
        # cut frames with the frequency e.g. step=8 or 16
        assert(frame_cnt > chunk_size)
        clipped_length = frame_cnt - chunk_size
        # the start of last chunk with stride=(8//4)=2
        clipped_length = (clipped_length // (frequency//4)) * (frequency//4)  
        # frames to clips
        # stride=2， e.g. the number of frames: 4096, clips: (4096-16)//2 + 1=2041
        frame_indices = []
        for i in range(4 * clipped_length // frequency + 1):
            frame_indices.append([j for j in range(i * frequency//4, i * frequency//4 + chunk_size)])
        frame_indices = np.array(frame_indices)

        # frame_indices = np.reshape(frame_indices, (-1, 16)) # Frames to chunks
        chunk_num = np.shape(frame_indices)[0]

        # Chunks to batches
        batch_num = int(np.ceil(chunk_num / batch_size))    
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        if sample_mode == 'oversample':
            full_features = [[] for i in range(10)]
        else:
            full_features = [[]]
        for batch_id in range(batch_num):
            
            require_resize = sample_mode == 'resize'
            # all pics or optical flow in one batch
            if mode == 'rgb':
                if usezip:
                    batch_data = load_ziprgb_batch(rgb_zipdata, rgb_files, frame_indices[batch_id], dataset, require_resize)
                else:                
                    batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id], dataset, require_resize)
            else:
                if usezip:
                    batch_data = load_zipflow_batch(
                        flow_x_zipdata, flow_y_zipdata,
                        flow_x_files, flow_y_files,
                        frame_indices[batch_id], require_resize)
                else:
                    batch_data = load_flow_batch(frames_dir, flow_x_files, flow_y_files, frame_indices[batch_id], dataset, require_resize)

            if sample_mode == 'oversample':
                batch_data_ten_crop = oversample_data(batch_data)

                for i in range(10):
                    pdb.set_trace()
                    assert(batch_data_ten_crop[i].shape[-2] == 224)
                    assert(batch_data_ten_crop[i].shape[-3] == 224)
                    full_features[i].append(forward_batch(batch_data_ten_crop[i]))

            else:
                if sample_mode == 'center_crop':
                    batch_data = batch_data[:, :, 16:240, 58:282, :]  # Centrer Crop  (39, 16, 224, 224, 2)
                
                assert(batch_data.shape[-2] == 224)
                assert(batch_data.shape[-3] == 224)
                full_features[0].append(forward_batch(batch_data))
            print('{}, {}/{}, completed.'.format(v_name, batch_num, (batch_id+1)))
        # all outputs in one video 
        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)
        # save all features from one video in save_path
        np.savez(os.path.join(subject_path, save_file),
                 feature=full_features,
                 frame_cnt=frame_cnt,
                 video_name=video_name)

        print('{} done: {} / {}, {}'.format(video_name, frame_cnt, clipped_length, full_features.shape))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='rgb') # flow or rgb
    parser.add_argument('--load_model', type=str, default='/home/yww/1_spot/pytorch-i3d-feature-extraction/models/rgb_imagenet.pt')
    parser.add_argument('--input_dir', type=str, default='/home/yww/1_spot/CAS(ME)2_longVideoFaceCropped/longVideoFaceCropped')
    parser.add_argument('--output_dir', type=str, default='/home/yww/WTAL/CAS_feature_4')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_mode', type=str, default='resize')
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cas')
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--merge',  default=False, action='store_true', help='merge videos or not')
    parser.add_argument('--usezip', dest='usezip', action='store_true')
    parser.add_argument('--no_usezip', dest='usezip', action='store_false')
    parser.set_defaults(no_usezip=True)

    args = parser.parse_args()
    print(args)
    run(dataset=args.dataset,
        mode=args.mode, 
        load_model=args.load_model,
        sample_mode=args.sample_mode,
        input_dir=args.input_dir, 
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        frequency=args.frequency,
        usezip=args.usezip,
        chunk_size=args.chunk_size,
        merge = args.merge)
