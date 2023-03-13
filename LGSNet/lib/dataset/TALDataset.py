import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TALDataset(Dataset):
    def __init__(self, cfg, split, subject):
        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.FEAT_DIR, subject)
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.target_size = (cfg.DATASET.RESCALE_TEM_LENGTH, cfg.MODEL.IN_FEAT_DIM)
        self.max_segment_num = cfg.DATASET.MAX_SEGMENT_NUM
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.base_dir = os.path.join(self.root, self.split)
        self.datas = self._make_dataset()

        self.class_label = cfg.DATASET.CLASS_IDX
        self.window_size = cfg.DATASET.WINDOW_SIZE
        if self.split == self.train_split:
            self.anno_df = pd.read_csv('/home/yww/1_spot/casme2_annotation.csv')

        self.gt_overlap_threshold = 0.9

    def __len__(self):
        return len(self.datas)

    def get_anno(self, start_frame, video_name):
        end_frame = start_frame + self.window_size

        label = list()
        box = list()
        anno_df = self.anno_df[self.anno_df.video == video_name]
        for i in range(len(anno_df)):
            act_start = anno_df.startFrame.values[i]
            act_end = anno_df.endFrame.values[i]
            assert act_end > act_start
            overlap = min(end_frame, act_end) - max(start_frame, act_start)
            overlap_ratio = overlap * 1.0 / (act_end - act_start)

            if overlap_ratio > self.gt_overlap_threshold:
                gt_start = max(start_frame, act_start) - start_frame
                gt_end = min(end_frame, act_end) - start_frame

                label.append(self.class_label.index(anno_df.type_idx.values[i]))
                box.append([gt_start, gt_end])  # frame level

        box = np.array(box).astype('float32')
        label = np.array(label)
        return label, box

    def __getitem__(self, idx):
        file_name = self.datas[idx]
        data = np.load(os.path.join(self.base_dir, file_name))

        feat_tem = data['feat_tem']
        # feat_tem = cv2.resize(feat_tem, self.target_size, interpolation=cv2.INTER_LINEAR)
        feat_spa = data['feat_spa']
        # feat_spa = cv2.resize(feat_spa, self.target_size, interpolation=cv2.INTER_LINEAR)
        begin_frame = data['begin_frame']
        # pass video_name vis list
        video_name = str(data['vid_name'])
        
        if self.split == self.train_split:
            action = data['action']
            # action_tmp = [i[:2] for i in action]
            action = np.array(action).astype('float32')
            label = data['class_label']
            # data for anchor-based
            # label, action = self.get_anno(begin_frame, video_name)
            num_segment = action.shape[0]
            assert num_segment > 0, 'no action in {}!!!'.format(video_name)
            action_padding = np.zeros((self.max_segment_num, 2), dtype=np.float)
            action_padding[:num_segment, :] = action
            label_padding = np.zeros(self.max_segment_num, dtype=np.int)
            label_padding[:num_segment] = label

            return feat_spa, feat_tem, action_padding, label_padding, num_segment
        else:
            return feat_spa, feat_tem, begin_frame, video_name

    def _make_dataset(self):
        datas = os.listdir(self.base_dir)
        datas = [i for i in datas if i.endswith('.npz')]
        return datas

