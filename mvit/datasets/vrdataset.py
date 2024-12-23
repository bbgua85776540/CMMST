import pickle
from torch.utils.data import Dataset
import json
import numpy as np

from sklearn import preprocessing
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Vrdataset(Dataset):

    def __init__(self, cfg, mode,dataset_flag=2, look_back=30, look_ahead=15):
        VIEW_PATH = 'F:/dataset/vr_dataset/ep1_head_base_frame_16x9/'

        # Get the necessary information regarding the dimensions of the video
        print("Dataset Reading JSON...")
        file = open('../mvit/datasets/meta.json', )
        jsonRead = json.load(file)
        # print('dataset_flag=', dataset_flag)
        self.nusers = jsonRead["dataset"][dataset_flag - 1]["nusers"]
        self.ntopics = jsonRead["dataset"][dataset_flag - 1]["ntopics"]
        frame_count = [4921, 5994, 8797, 5172, 6165, 19632, 11251, 4076, 8603]
        video_time = [164, 201, 293, 172, 205, 655, 451, 164, 292]
        self.time_start = [7, 5, 80, 15, 45, 152, 25, 86, 35]
        self.time_end = [45, 35, 130, 45, 75, 182, 45, 116, 80]
        self.frame_start = []
        self.frame_end = []
        for idx in range(len(frame_count)):
            start = int(frame_count[idx]/video_time[idx] * self.time_start[idx])
            end = int(frame_count[idx] / video_time[idx] * self.time_end[idx])
            self.frame_start.append(start)
            self.frame_end.append(end)
        self.frame_count = [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600]
        self.width = jsonRead["dataset"][dataset_flag - 1]["width"]
        self.height = jsonRead["dataset"][dataset_flag - 1]["height"]
        self.view_width = jsonRead["dataset"][dataset_flag - 1]["view_width"]
        self.view_height = jsonRead["dataset"][dataset_flag - 1]["view_height"]
        self.milisec = jsonRead["dataset"][dataset_flag - 1]["milisec"]
        self.look_back = cfg.fps
        self.n_col = cfg.n_colum
        self.n_row = cfg.n_row
        self.process_frame_nums = cfg.process_frame_nums
        self.file_list = []
        for topic in range(self.ntopics):  #  0 1 2 3 4 5

            series_start = self.frame_start[topic]
            series_num = self.frame_end[topic]

            if topic > 4: continue
            if topic == 6 or topic == 7:   # sal _29fps   实际上是25  没转
                topic_ = 'topic{}/'.format(topic)
                topic_sal = '{}_29fps/'.format(topic + 1)
            else:
                topic_ = 'topic{}/'.format(topic)
                topic_sal = '{}/'.format(topic + 1)

            print('topic=', topic)
            for user in range(48):
                train_val_index = 0
                for series_index in range(series_start, series_num - self.process_frame_nums, 10):
                        view_file_path = VIEW_PATH + topic_ + 'headmaps_topic{}_user{}_'.format(topic, user + 1)
                        label_file_path = VIEW_PATH + topic_ + 'labelmaps_topic{}_user{}_'.format(topic, user + 1)
                        # sal_file_path = '/media/kemove/DE762BE9762BC15F/dataset/vr_dataset_video/sal/{}_sal_256x144'.format(
                        #     topic + 1)
                        sal_file_path = 'E:/dataset/vr_dataset_video/sal/back0319/' + topic_sal
                        self.file_list.append([view_file_path, label_file_path, sal_file_path, topic, series_index])



    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.file_list)

    def __getitem__(self, index):
        # print('index=', index)
        headmaps = []
        labelmaps = []
        # print('end load view_info')
        sal_info = []
        sal_file_path = self.file_list[index][2]
        series_index = self.file_list[index][4]
        topic = self.file_list[index][3]
        start = series_index
        end = series_index + self.process_frame_nums
        for frame_index in range(start, end):
            # path = sal_file_path + '{}_sal_256x144.pkl'.format(frame_index)
            if topic == 6 or topic == 7:
                # print(topic)
                path = sal_file_path + '{}_sal_256x144_29fps.pkl'.format(frame_index)
                # path = sal_file_path + '{}_sal_256x144.pkl'.format(frame_index)
            else:
                path = sal_file_path + '{}_sal_256x144.pkl'.format(frame_index)
            sal_info.append(pickle.load(open(path, 'rb'), encoding='latin1'))
            if frame_index < start + self.look_back:
                headmap_path = self.file_list[index][0] + 'series{}_256x144.pkl'.format(frame_index)
                headmaps.append(pickle.load(open(headmap_path, 'rb'), encoding='latin1'))
            labelmap_path = self.file_list[index][1] + 'series{}_256x144.pkl'.format(frame_index)
            labelmaps.append(pickle.load(open(labelmap_path, 'rb'), encoding='latin1'))
        sal_info = np.array(sal_info).astype('float32')
        # sal_info = sal_info/255
        sal_info = sal_norm(sal_info)
        headmaps = np.array(headmaps).astype('float32')
        labelmaps = np.array(labelmaps).astype('float32')
        # return headmaps, headmaps, labelmaps
        return headmaps, sal_info, labelmaps, series_index

def sal_norm(my_arr):
    my_min_val = np.min(my_arr)
    my_max_val = np.max(my_arr)

    # Perform min-max normalization
    my_normalized_arr = (my_arr - my_min_val) / (my_max_val - my_min_val)
    return my_normalized_arr

def get_frame_pos(frame_nos, headmaps, frame_nums, labelmaps):
    frame_pos = []
    label_pos = []
    # print('frame_nos=', frame_nos)
    # print('frame_nos[119]=', frame_nos[119])
    # print('frame_nums=', frame_nums)
    for i in range(frame_nums):
        pos_index = np.where(frame_nos == i)
        # print('pos_index=', pos_index)
        j = i
        # print('len(pos_index)=', len(pos_index))
        while len(pos_index[0]) == 0:
            j += 1
            pos_index = np.where(frame_nos == j)
            print('do offset=', j - i)
        # print('pos_index=', pos_index)
        pos_index = pos_index[0]
        # print('pos_index=', pos_index)
        pos = headmaps[pos_index[0]]
        # print('i=', i)
        # print('pos=', pos)
        frame_pos.append(pos)
        label_pos.append(labelmaps[pos_index[0]])
    return frame_pos, label_pos
