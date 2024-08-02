import numpy as np 
import math
import sys
import pickle
import argparse
import json
import numpy
import sys
sys.path.append('/CMMST')
# from mvit.datasets.utils import get_act_tiles3
# from mvit.datasets.vrdataset import get_frame_pos
# from visualization_viewport import get_frame_pos

def main():

	parser = argparse.ArgumentParser(description='Calculate QoE and error for PanoSalNet algorithm')

	parser.add_argument('-D', '--dataset', type=int, default=2, help='Dataset ID (1 or 2)')
	parser.add_argument('-T', '--topic', default=8, help='Topic in the particular Dataset (video name)')
	parser.add_argument('--fps', type=int, default=29, help='fps of the video')
	parser.add_argument('-Q', '--quality', default='360p', help='Preferred bitrate quality of the video (360p, 480p, 720p, 1080p, 1440p)')
	parser.add_argument('--pred_time', type=int, default=30, help='predict time of the video')

	args = parser.parse_args()

	PATH_PRED = 'F:/mvit-result/186/CMMST/result/pred_viewport/'

	if args.dataset != 1 and args.dataset != 2:
		print("Incorrect value of the Dataset ID provided!!...")
		print("======= EXIT ===========")
		exit()

	# Get the necessary information regarding the dimensions of the video
	print("Reading JSON...")
	file = open('/CMMST/mvit/datasets/meta.json', )
	jsonRead = json.load(file)

	ncol_tiles = 16
	nrow_tiles = 9


	manhattan_error, x_mae, y_mae, final_qoe = [],[],[],[]
	count_frames = 0
	nusers = 48
	if args.topic == 6 or args.topic == 7:
		# topic_ = 'topic{}_29fps/'.format(args.topic)
		topic_ = 'topic{}/'.format(args.topic)
	else:
		topic_ = 'topic{}/'.format(args.topic)
	correct_num = 0
	total_num = 0

	frame_count = [4921, 5994, 8797, 5172, 6165, 19632, 11251, 4076, 8603]
	video_time = [164, 201, 293, 172, 205, 655, 451, 164, 292]
	time_start = [7, 5, 80, 15, 45, 152, 25, 86, 35]
	time_end = [45, 35, 130, 45, 75, 182, 45, 116, 80]
	frame_start = []
	frame_end = []
	for idx in range(len(frame_count)):
		start = int(frame_count[idx] / video_time[idx] * time_start[idx])
		end = int(frame_count[idx] / video_time[idx] * time_end[idx])
		frame_start.append(start)
		frame_end.append(end)
	series_start = frame_start[args.topic]
	series_num = frame_end[args.topic]

	for usernum in range(0, 48):
		# print('User_{}'.format(usernum))
		# if usernum == 11: continue
		user_manhattan_error = 0.
		view_file_path = 'F:/dataset/vr_dataset/ep1_head_base_frame_16x9/' + \
						  topic_+ 'labelmaps_topic{}_user{}_'.format(args.topic, usernum + 1)
		# viewport = pickle.load(open(VIEW_PATH + "viewport_ds{}_topic{}_user{}".format(args.dataset, args.topic, usernum+1), "rb"), encoding='latin1')
		p_viewport = pickle.load(open(PATH_PRED + "ds{}_topic{}_user{}".format(args.dataset, args.topic, usernum+1), "rb"), encoding="latin1")
		viewport = []
		for index in range(series_start, series_num):
			headmap_path = view_file_path + 'series{}_256x144.pkl'.format(index)
			headmap = pickle.load(open(headmap_path, 'rb'), encoding='latin1')
			headmap = headmap.reshape(ncol_tiles, nrow_tiles)
			vp = np.where(headmap == headmap.max())
			viewport.append((vp[0][0], vp[1][0]))

		act_viewport = np.array(viewport)
		# print('act_viewport=', act_viewport.shape)    # act_viewport = (1800, 9, 16)
		# print('p_viewport=', len(p_viewport))  # act_viewport = 1749

		# Predicted Tile = max of the probabilities in output
		pred_max_viewport = []
		p_viewport = numpy.concatenate(p_viewport, axis=0)
		# for frs in range(len(p_viewport)):
		for frs in range(len(p_viewport)):
			# print('p_viewport[frs].shape=', p_viewport[frs].shape)    # torch.Size([9, 16])
			prob = p_viewport[frs]
			argmax = np.where(prob == prob.max())
			# print(frs, 'user=', usernum,'argmax=', argmax)
			pred_max_viewport.append((argmax[0][0], argmax[1][0]))
		# print('pred_max_viewport=', len(pred_max_viewport))          # pred_max_viewport= 1749


		# Assert len(actual frames) = len(predicted frames)
		pred_viewport = p_viewport
		act_viewport = act_viewport[args.pred_time:args.pred_time+len(pred_viewport)]

		# pred_max_viewport = pred_max_viewport[120:]
		# act_viewport = act_viewport[120:]
		# print('act_viewport=', act_viewport.shape)  # act_viewport= (1749, 9, 16)
		# frame_nos = frame_nos[args.pred_time:args.pred_time+len(pred_viewport)]
		# act_viewport = act_viewport[:len(pred_viewport)]
		# frame_nos = frame_nos[:len(pred_viewport)]

		# pred_viewport = pred_viewport[:len(act_viewport)]
		# frame_nos = frame_nos[:len(pred_viewport)]
		# print('pred_viewport=', len(pred_viewport))
		# print('act_viewport=', len(act_viewport))
		# print('pred_max_viewport=', len(act_viewport))

		pred_max_viewport = np.array(pred_max_viewport)
		# Calculate Manhattan Error
		for fr in range(len(pred_max_viewport)):
			act_tile = act_viewport[fr]
			pred_tile = pred_max_viewport[fr]
			total_num += 1
			if (act_tile == pred_tile).all():
				correct_num+=1
			print('act_tile=', act_tile, ' pred_tile=', pred_tile)
			# Get corrected error
			# tile_col_dif = ncol_tiles
			tile_row_dif = act_tile[0] - pred_tile[0]
			tile_col_dif = min(pred_tile[1]-act_tile[1], act_tile[1]+ncol_tiles-pred_tile[1]) if act_tile[1] < pred_tile[1] else min(act_tile[1]-pred_tile[1], ncol_tiles+pred_tile[1]-act_tile[1])

			current_tile_error = abs(tile_row_dif) + abs(tile_col_dif)
			user_manhattan_error += current_tile_error


		manhattan_error.append(user_manhattan_error/len(pred_max_viewport))
		count_frames += len(act_viewport)

	# avg_qoe = np.mean(final_qoe)
	avg_manhattan_error = np.mean(manhattan_error)

	#Print averaged results
	print("\n======= RESULTS ============")
	print('PanoSalNet')
	print('Dataset: {}'.format(args.dataset))
	print('Topic: ' + str(args.topic))
	print('Pred_nframe: {}'.format(args.fps))
	# print('Avg. QoE: {}'.format(avg_qoe))
	print('Avg. Manhattan error: {}'.format(avg_manhattan_error))
	print('Count: {}'.format(count_frames))
	print('precition:{}  and correct_num={}, total_num={}'.format(correct_num/total_num, correct_num, total_num))
	print('\n\n')


if __name__ == "__main__":
	main()
