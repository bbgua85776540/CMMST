import numpy as np
import math
import pickle


def get_act_tiles(dataset, view_info, frame_nos, fps, milisec, width, height, view_width, view_height, ncol_tiles, nrow_tiles):
    """
	Calculate the tiles corresponding to the viewport
	"""
    act_viewport = []
    headmaps = []
    max_frame = int(view_info[-1][0] * 1.0 * fps / milisec)
    for i in range(len(view_info) - 1):
        frame = int(view_info[i][0] * 1.0 * fps / milisec)
        frame_nos.append(frame)
        if (frame > max_frame):
            break
        view_x = int(view_info[i][1][1] * width / view_width)
        view_y = int(view_info[i][1][0] * height / view_height)
        # print('view_x=', view_x, '  view_y=', view_y)
        # dataset 2
        if dataset == 2:
            view_x = width - view_x
            view_y = height - view_y
        tile_col = int(view_x * ncol_tiles / width)
        tile_row = int(view_y * nrow_tiles / height)
        # print('tile_col=', tile_col, '  tile_row=', tile_row)
        if tile_col >= ncol_tiles:
            print('tile_col--', tile_col)
            tile_col = ncol_tiles - 1
        if tile_row >= nrow_tiles:
            print('tile_row--', tile_row)
            tile_row = nrow_tiles - 1

        # 0222 生成map
        headmap = np.zeros(shape=(nrow_tiles, ncol_tiles))
        headmap[tile_row, tile_col] = 1
        headmaps.append(headmap)
        act_viewport.append((tile_row, tile_col))

    return act_viewport, frame_nos, max_frame, headmaps


def get_act_tiles2(dataset, view_info, fps, milisec, width, height, view_width, view_height, ncol_tiles, nrow_tiles):
    #  仅仅获取  max_frame, headmaps
    headmaps = []
    max_frame = int(view_info[-1][0] * 1.0 * fps / milisec)
    for i in range(len(view_info) - 1):
        view_x = view_info[i][1][1] * width / view_width
        view_y = view_info[i][1][0] * height / view_height
        # dataset 2
        if dataset == 2:
            view_x = width - view_x
            view_y = height - view_y
        tile_col_deci = view_x * ncol_tiles / width
        tile_col = int(tile_col_deci)
        tile_row_deci = view_y * nrow_tiles / height
        tile_row = int(tile_row_deci)
        headmap = np.zeros(shape=(nrow_tiles, ncol_tiles))
        headmap[tile_row, tile_col] = 1
        headmaps.append(headmap)
    return max_frame, headmaps


def get_act_tiles3(dataset, view_info, frame_nos, fps, milisec, width, height, view_width, view_height, ncol_tiles, nrow_tiles):
    """
	Calculate the tiles corresponding to the viewport
	"""
    act_viewport = []
    max_frame = int(view_info[-1][0] * 1.0 * fps / milisec)

    for i in range(len(view_info) - 1):
        frame = int(view_info[i][0] * 1.0 * fps / milisec)
        frame_nos.append(frame)
        if (frame > max_frame):
            break

        view_x = int(view_info[i][1][1] * width / view_width)
        view_y = int(view_info[i][1][0] * height / view_height)
        #dataset 2
        if dataset == 2:
            view_x = width - view_x
            view_y = height - view_y
        # tile_col = ncol_tiles - int(view_x * ncol_tiles / width) - 1
        # tile_row = nrow_tiles - int(view_y * nrow_tiles / height) - 1
        tile_col = int(view_x * ncol_tiles / width)
        tile_row = int(view_y * nrow_tiles / height)

        act_viewport.append((tile_row, tile_col))

    return act_viewport, frame_nos
