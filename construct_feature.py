import argparse
import numpy as np
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('-phase', type=str, default='train')
parser.add_argument('-tem_feature_dir', type=str, default='/home/yww/1_spot/CA_mag5_flow_feature')
parser.add_argument('-spa_feature_dir', type=str, default='/home/yww/1_spot/CA_mag5_rgb_feature')
parser.add_argument('-wind_info_file', type=str, default='./window_info.log')
parser.add_argument('-gt_box_file', type=str, default='./gt_box.pkl')
parser.add_argument('-gt_label_file', type=str, default='./gt_label.pkl')
parser.add_argument('-res_dir', type=str, default='./CA_mag5_feature/train')
parser.add_argument('-res_dir_test', type=str, default='./CA_mag5_feature/test')
parser.add_argument('-sample_interval', type=int,  default=2, help='sampling rate in raw video')
parser.add_argument('-wind_length', type=int, default=64, help='sliding window//sample_interval')
args = parser.parse_args()


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]  # ((all frames in one video) /frequency) x feature_dim(1024)
    return feature


def pad_feature(feature, feat_win):
    t = feature.shape[0]
    feature_slice = feature[-1:, :]
    for _ in range(t, feat_win):
        feature = np.concatenate((feature, feature_slice), axis=0)
    assert feature.shape[0] == feat_win
    return feature


def construct_feature(vid_name, feat_file, start_idx, end_idx, wind_length):
    feature = load_npz_feat(feat_file)
    end_tmp = min(end_idx, feature.shape[0])
    feat_sel = feature[start_idx:end_tmp, :]

    if end_idx >= feature.shape[0] + 1:
        print(end_idx, feature.shape[0] + 1)
        print('%s padding feature, original length: %d, desired length: %d' % (vid_name, feature.shape[0], end_idx))
        feat_sel = pad_feature(feat_sel, feat_win=wind_length)
    feat = np.transpose(feat_sel)
    return feat


if __name__ == '__main__':
    
    with open(args.wind_info_file, 'r') as f:
        lines = f.readlines()
    if args.phase == 'train':
        gt_box = pickle.load(open(args.gt_box_file, 'rb'))
        gt_label = pickle.load(open(args.gt_label_file, 'rb'))

    for iord, line in enumerate(lines):
        begin_frame, vid_name = line.split(',')
        vid_name = vid_name[1:-1]
        save_file = vid_name + '_' + begin_frame.zfill(5) + '.npz'

        if args.phase == 'train':
            info = gt_box[iord]
            label = gt_label[iord]
        else:
            info = None
            label = None

        start_idx = int(int(begin_frame) / args.sample_interval)
        end_idx = start_idx + args.wind_length

        mode = 'flow'
        feat_file = os.path.join(args.tem_feature_dir, vid_name + '-' + mode + '.npz')
        feat_tem = construct_feature(vid_name, feat_file, start_idx, end_idx)

        mode = 'rgb'
        feat_file = os.path.join(args.spa_feature_dir, vid_name + '-' + mode + '.npz')
        feat_spa = construct_feature(vid_name, feat_file, start_idx, end_idx)

        if args.phrase == 'train':
            if not os.path.exists(args.res_dir):
                os.makedirs(args.res_dir)
        else:
            args.res_dir = args.res_dir_test
            if not os.path.exists(args.res_dir):
                os.makedirs(args.res_dir)

        if os.path.exists(os.path.join(args.res_dir, save_file)):
            print('already exists', line)

        np.savez(os.path.join(args.res_dir, save_file),
                 vid_name=vid_name,
                 begin_frame=int(begin_frame),
                 action=info,
                 class_label=label,
                 feat_tem=feat_tem,
                 feat_spa=feat_spa)