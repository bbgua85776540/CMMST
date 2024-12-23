import numpy as np
import math
import sys
import pickle
import argparse
import json
import sys

sys.path.append('/CMMST')
from qoe import calc_qoe
from mvit.datasets.utils import get_act_tiles, get_act_tiles2
from mvit.datasets.vrdataset import get_frame_pos
import cv2
from sklearn import preprocessing
import os


def main():
    parser = argparse.ArgumentParser(description='Calculate QoE and error for PanoSalNet algorithm')

    parser.add_argument('-D', '--dataset', type=int, default=2, help='Dataset ID (1 or 2)')
    parser.add_argument('-T', '--topic', default=8, help='Topic in the particular Dataset (video name)')
    parser.add_argument('--fps', type=int, default=29, help='fps of the video')
    parser.add_argument('-Q', '--quality', default='360p',
                        help='Preferred bitrate quality of the video (360p, 480p, 720p, 1080p, 1440p)')
    parser.add_argument('--pred_time', type=int, default=50, help='predict time of the video')

    args = parser.parse_args()

    if args.dataset != 1 and args.dataset != 2:
        print("Incorrect value of the Dataset ID provided!!...")
        print("======= EXIT ===========")
        exit()

    # Get the necessary information regarding the dimensions of the video
    print("Reading JSON...")
    file = open('/CMMST/mvit/datasets/meta.json', )
    jsonRead = json.load(file)

    width = jsonRead["dataset"][args.dataset - 1]["width"]
    height = jsonRead["dataset"][args.dataset - 1]["height"]
    view_width = jsonRead["dataset"][args.dataset - 1]["view_width"]
    view_height = jsonRead["dataset"][args.dataset - 1]["view_height"]
    milisec = jsonRead["dataset"][args.dataset - 1]["milisec"]
    # frame_count = [4921, 5994, 8797, 5172, 6165, 19632, 11251, 4076, 8603]
    frame_count = [4921, 5994, 8797, 5172, 6165, 19632, 11251, 4076, 8602]
    video_time = [164, 201, 293, 172, 205, 655, 451, 164, 292]
    # fps = [29, 29, 30, 29, 29, 29, 25, 25, 29]
    # fps = [29, 29, 30, 29, 29, 29, 29, 29, 29]    # 将topic 7  8  转成了29帧
    gblur_size = 9
    n_col = 16
    n_row = 9
    mmscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    VIEW_PATH = 'E:/work/pytorch_workplace/PARIMA-master/Viewport/ds2/'
    # out_dir = 'E:/dataset/vr_dataset/ep1_head_base_frame/'
    out_dir = 'F:/dataset/vr_dataset/ep1_head_base_frame_16x9/'

    nusers = 48

    #每个step生成一个文件
    for topic in range(0, 9):
        # if topic != 8 : continue
        print('topic=', topic)
        out_dir_topic = out_dir + 'topic{}/'.format(topic)
        if not os.path.exists(out_dir_topic):
            os.makedirs(out_dir_topic)
        for usernum in range(nusers):
            # if usernum != 39: continue
            print('user=', usernum)
            viewport = pickle.load(
                open(VIEW_PATH + "viewport_ds{}_topic{}_user{}".format(args.dataset, topic, usernum + 1), "rb"),  ##  0319修正之前的 args.topic 导致的问题
                encoding='latin1')

            frame_nos = []
            fps = frame_count[topic]/video_time[topic]
            act_viewport, frame_nos, max_frame, headmaps = get_act_tiles(args.dataset, viewport, frame_nos, fps,
                                                                         milisec, width, height,
                                                                         view_width, view_height, 256,
                                                                         144)
            max_frame, labelmaps = get_act_tiles2(args.dataset, viewport, fps,
                                                                         milisec, width, height,
                                                                         view_width, view_height, n_col,
                                                                         n_row)
            headmaps, labelmaps = get_frame_pos(np.array(frame_nos), headmaps, frame_count[topic], labelmaps)
            headmaps = np.array(headmaps)
            headmaps = np.array([cv2.GaussianBlur(item, (gblur_size, gblur_size), 0) for item in headmaps])
            headmaps = mmscaler.fit_transform(headmaps.ravel().reshape(-1, 1)).reshape(headmaps.shape)
            headmaps = np.array(headmaps).astype('float16')

            labelmaps = np.array(labelmaps).astype('uint8')
            labelmaps = labelmaps.reshape(len(labelmaps), n_col * n_row)
            for series_index in range(len(headmaps)):
                # start_index = series_index
                # end_index = series_index + process_frame_nums
                headmaps_ser = headmaps[series_index]
                labelmaps_ser = labelmaps[series_index]
                pickle.dump(headmaps_ser, open(out_dir_topic + 'headmaps_topic{}_user{}_series{}_256x144.pkl'.format(topic, usernum + 1, series_index), 'wb'))
                pickle.dump(labelmaps_ser, open(out_dir_topic + 'labelmaps_topic{}_user{}_series{}_256x144.pkl'.format(topic, usernum + 1, series_index), 'wb'))

if __name__ == "__main__":
    main()
