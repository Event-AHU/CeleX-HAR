import os
import pdb
import csv
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io
from spconv.pytorch.utils import PointToVoxel


def transform_points_to_voxels(data_dict={}, device=torch.device("cuda:0")):
    """
    将点云转换为voxel,调用spconv的VoxelGeneratorV2
    """
    points = data_dict['points']
    # 将points打乱
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    data_dict['points'] = points

    voxel_generator = PointToVoxel(
        vsize_xyz=[4, 4, 4],  # [0.2, 0.25, 0.16]

        coors_range_xyz=[0, 0, 0, 127, 127, 127],

        num_point_features=4,

        max_num_voxels=5000,  # 16000

        max_num_points_per_voxel=16,  # 32
        device=device
    )

    points = torch.tensor(data_dict['points'])
    points = points.to(device)
    voxel_output = voxel_generator(points)

    voxels, coordinates, num_points = voxel_output
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # 选event数量在前5000的voxel
    if num_points.shape[0] < 1024:
        features = voxels[:, :, 3]
        coor = coordinates
    else:
        _, voxels_idx = torch.topk(num_points, 1024)
    # 将每个voxel的64个p拼接作为voxel初始特征
        features = voxels[voxels_idx][:, :, 3]
    # 前2048个voxel的三维坐标
        coor = coordinates[voxels_idx]
    # 将y.x.t改为t,x,y
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]


    return coor, features


if __name__ == '__main__':
    data_path = '/csv'
    save_path = '/voxel'
    device = torch.device("cuda:0")
    
    # for train_test in os.listdir(data_path):
    fileLIST = os.listdir(os.path.join(data_path))
    for cls_ID in range(len(fileLIST)):
        cls = fileLIST[cls_ID]
        # fileLIST = os.listdir(os.path.join(data_path,train_test,cls))
        if cls.endswith("csv"):
            # save_dir = os.path.join(save_path, os.path.splitext(cls)[0])
            # # breakpoint()
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            for FileID in tqdm(range(len(fileLIST))):
                file_Name = fileLIST[FileID]
                read_path = os.path.join(data_path, cls)
                save_name = file_Name.split('.')[0]
                mat_save = os.path.join(save_path,'{}.mat'.format(save_name))
                if os.path.exists(mat_save):
                    continue
                with open(read_path, 'r') as csv_path:
                    dt = csv.reader(csv_path)
                    data_list = [list(map(int, row)) for row in dt]
                    dt = np.array(data_list)
                    dt = torch.tensor(dt, dtype=torch.int)
                    dt.to(device)
                    # t, x, y, p = torch.chunk(dt, 4, dim=1)
                    x,y,p,t = torch.chunk(dt, 4, dim=1)
                    
                    # get the last timestamp
                    time_length = t[-1]-t[0]
                    # rescale the timestampes to start from 0 up to 128
                    t = (((t-t[0])/ time_length)*127).float()

                    mask = torch.where(p == 0)
                    p[mask] = -1
                    all_events = torch.cat((x,y,p,t), dim=1)
                    # all_events = torch.cat((t, x, y, p), dim=1)
                    data_dict = {'points': all_events}
                    coor, features = transform_points_to_voxels(data_dict=data_dict, device=device)
                    coor = coor.cpu()
                    features = features.cpu()
                    coor = coor.numpy()
                    features = features.numpy()
                    scipy.io.savemat(mat_save, mdict={'coor': coor, 'features': features})
