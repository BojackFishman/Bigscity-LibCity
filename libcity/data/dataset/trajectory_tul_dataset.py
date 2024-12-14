import os
import json
import pandas as pd
import math
from tqdm import tqdm
import importlib
from logging import getLogger

from libcity.data.dataset import AbstractDataset
from libcity.utils import parse_time, cal_timeoff
from libcity.data.utils import generate_dataloader_pad

from libcity.utils.rawprocess import grid_process, generate_graph
import torch

parameter_list = ['dataset', 'min_session_len', 'min_sessions', "max_session_len",
                  'cut_method', 'window_size', 'min_checkins']
graph_distinct_list = ['train_rate', 'eval_rate']

class TrajectoryTulDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        self.cut_data_cache = './libcity/cache/dataset_cache/cut_traj'
        for param in parameter_list:
            self.cut_data_cache += '_' + str(self.config[param])
        self.cut_data_cache += '.json'
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.data = None
        # 加载 encoder
        self.encoder = self.get_encoder()
        self.pad_item = None  # 因为若是使用缓存, pad_item 是记录在缓存文件中的而不是 encoder
        self.logger = getLogger()

        # 图数据的cache路径
        self.graph_data_cache = './libcity/cache/dataset_cache/graph'
        for param in parameter_list:
            self.graph_data_cache += '_' + str(self.config[param])
        for param in graph_distinct_list:
            self.graph_data_cache += '_' + str(self.config[param])
        self.graph_data_cache += '.pt'      # graph存储为.pt格式
        
        # 添加数据
        self.grid_list = None   # 网格id列表
        self.traj_list = None   # 轨迹id列表
        self.user_list = None   # 用户列表
        self.grid_nums = None
        self.user_nums = None
        # self.user_traj_dict = {}     # 用户轨迹数据，字典
        # self.user_traj_train = {}   # 用户训练数据，字典

        self.local_feature = None
        self.local_adj = None
        self.global_feature = None
        self.global_adj = None

        self.user_traj_train = None     # 字典形式的训练数据

    def get_data(self):
        """
        轨迹比较特殊，原子文件中存储的并不是轨迹而是一个一个点，因此需要先对轨迹进行切割
        """
        if self.data is None:
            if self.config['cache_dataset'] and os.path.exists(self.encoder.cache_file_name):
                # load cache
                f = open(self.encoder.cache_file_name, 'r')
                self.data = json.load(f)
                self.pad_item = self.data['pad_item']
                self.grid_list = self.data['grid_list']
                self.traj_list = self.data['traj_list']
                self.user_list = self.data['user_list']
                self.grid_nums = self.data['grid_nums']
                self.user_nums = self.data['user_nums']
                f.close()
            else:
                if os.path.exists(self.cut_data_cache):
                    f = open(self.cut_data_cache, 'r')
                    cut_data = json.load(f)
                    f.close()
                else:
                    cut_data = self.cutter_filter()
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.cut_data_cache, 'w') as f:
                        json.dump(cut_data, f)
                self.logger.info('finish cut data')
                encoded_data = self.encode_traj(cut_data)
                self.data = encoded_data
                self.pad_item = self.encoder.pad_item
                if self.config['cache_dataset']:
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.encoder.cache_file_name, 'w') as f:
                        json.dump(encoded_data, f)

        # cut_data = self.cutter_filter()
        # encoded_data = self.encode_traj(cut_data)
        # self.data = encoded_data
        # self.pad_item = self.encoder.pad_item
        
        # user 来划，以及按轨迹数来划。
        # TODO: 这里可以设一个参数，现在先按照轨迹数来划吧
        train_data, eval_data, test_data = self.divide_data()


        if self.config['cache_dataset'] and os.path.exists(self.graph_data_cache):
            graph_data = torch.load(self.graph_data_cache)
            self.local_feature = graph_data['local_feature']
            self.local_adj = graph_data['local_adj']
            self.global_feature = graph_data['global_feature']
            self.global_adj = graph_data['global_adj']
        else:
            # 创建局部和全局特征图
            self.local_feature, self.local_adj, self.global_feature, self.global_adj = generate_graph(
                self.grid_list, self.traj_list, self.user_list, self.data['encoded_data'], self.user_traj_train)
            if self.config['cache_dataset']:
                graph_data = {}
                graph_data['local_feature'] = self.local_feature
                graph_data['local_adj'] = self.local_adj
                graph_data['global_feature'] = self.global_feature
                graph_data['global_adj'] = self.global_adj
                if not os.path.exists(self.cache_file_folder):
                    os.makedirs(self.cache_file_folder)
                torch.save(graph_data, self.graph_data_cache)

        return generate_dataloader_pad(train_data, eval_data, test_data,
                                       self.encoder.feature_dict,
                                       self.config['batch_size'],
                                       self.config['num_workers'], self.pad_item,
                                       self.encoder.feature_max_len)

    def get_data_feature(self):
        res = self.data['data_feature']
        res['local_feature'] = self.local_feature
        res['local_adj'] = self.local_adj
        res['global_feature'] = self.global_feature
        res['global_adj'] = self.global_adj
        res['grid_nums'] = self.grid_nums
        res['user_nums'] = self.user_nums
        return res

    def cutter_filter(self):
        """
        切割后的轨迹存储格式: (dict)
            {
                uid: [
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        # traj = pd.read_csv(os.path.join(
        #     self.data_path, '{}.dyna'.format(self.dyna_file)))
        traj, self.grid_list = self.gen_traj()
        self.grid_nums =len(self.grid_list)

        # filter inactive poi
        group_location = traj.groupby('location').count()
        filter_location = group_location[group_location['time'] >= self.config['min_checkins']]
        location_index = filter_location.index.tolist()
        traj = traj[traj['location'].isin(location_index)]
        user_set = pd.unique(traj['entity_id'])

        self.user_list = user_set.tolist()
        self.user_nums = len(self.user_list)

        # geo = pd.read_csv(os.path.join(
        #     self.data_path, '{}.geo'.format(self.geo_file)))
        
        # 在traj中添加一列轨迹索引traj_id
        traj['traj_id'] = 0

        res = {}
        min_session_len = self.config['min_session_len']
        max_session_len = self.config['max_session_len']
        min_sessions = self.config['min_sessions']
        window_size = self.config['window_size']
        cut_method = self.config['cut_method']
        if cut_method == 'time_interval':
            # 按照时间窗口进行切割
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    if index == 0:
                        session.append(row.tolist())
                        prev_time = now_time
                    else:
                        time_off = cal_timeoff(now_time, prev_time)
                        if time_off < window_size and time_off >= 0 and len(session) < max_session_len:
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            session.append(row.tolist())
                    prev_time = now_time
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        elif cut_method == 'same_date':
            # 将同一天的 check-in 划为一条轨迹
            traj_id = 0
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                prev_date = None
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    now_date = now_time.day
                    if index == 0:
                        session.append(row.tolist())
                    else:
                        if prev_date == now_date and len(session) < max_session_len:
                            # 还是同一天
                            row[-1] = traj_id   # traj_id不变，添加traj_id
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                traj_id = traj_id+1
                                sessions.append(session)
                            session = []
                            # traj_id = traj_id+1
                            row[-1] = traj_id
                            session.append(row.tolist())
                    prev_date = now_date
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
            self.traj_list = list(range(traj_id+1))     # 生成轨迹id列表
        else:
            # cut by fix window_len used by STAN
            if max_session_len != window_size:
                raise ValueError('the fixed length window is not equal to max_session_len')
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    if len(session) < window_size:
                        session.append(row.tolist())
                    else:
                        sessions.append(session)
                        session = []
                        session.append(row.tolist())
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        return res

    def encode_traj(self, data):
        """encode the cut trajectory

        Args:
            data (dict): the key is uid, the value is the uid's trajectories. For example:
                {
                    uid: [
                        trajectory1,
                        trajectory2
                    ]
                }
                trajectory1 = [
                    checkin_record,
                    checkin_record,
                    .....
                ]

        Return:
            dict: For example:
                {
                    data_feature: {...},
                    pad_item: {...},
                    encoded_data: {uid: encoded_trajectories}
                }
        """
        encoded_data = {}
        for uid in tqdm(data, desc="encoding trajectory"):
            encoded_data[int(uid)] = self.encoder.encode(int(uid), data[uid])
        self.encoder.gen_data_feature()
        return {
            'data_feature': self.encoder.data_feature,
            'pad_item': self.encoder.pad_item,
            'grid_list': self.grid_list,
            'traj_list': self.traj_list,
            'user_list': self.user_list,
            'grid_nums': self.grid_nums,
            'user_nums': self.user_nums,
            'encoded_data': encoded_data
        }

    def divide_data(self):
        """
        return:
            train_data (list)
            eval_data (list)
            test_data (list)
        """
        train_data = []
        eval_data = []
        test_data = []
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        user_set = self.data['encoded_data'].keys()

        user_traj_train = {key: [] for key in user_set}

        for uid in tqdm(user_set, desc="dividing data"):
            encoded_trajectories = self.data['encoded_data'][uid]
            traj_len = len(encoded_trajectories)
            # 根据 traj_len 来划分 train eval test
            train_num = math.ceil(traj_len * train_rate)
            eval_num = math.ceil(
                traj_len * (train_rate + eval_rate))
            train_data += encoded_trajectories[:train_num]
            user_traj_train[uid] += encoded_trajectories[:train_num]
            eval_data += encoded_trajectories[train_num:eval_num]
            test_data += encoded_trajectories[eval_num:]
        
        self.user_traj_train = user_traj_train
        return train_data, eval_data, test_data

    def get_encoder(self):
        try:
            return getattr(importlib.import_module('libcity.data.dataset.trajectory_encoder'),
                           self.config['traj_encoder'])(self.config)
        except AttributeError:
            raise AttributeError('trajectory encoder is not found')

    def gen_traj(self):
        # traj = pd.read_csv(os.path.join(
        #     self.data_path, '{}.dyna'.format(self.dyna_file)))
        
        traj = pd.read_csv(os.path.join(
            self.data_path, '{}.dyna'.format(self.dyna_file)))        
        geo = pd.read_csv(os.path.join(
            self.data_path, '{}.geo'.format(self.geo_file)))
        
        # print(geo.head(3))
        # print(geo.loc[2, 'coordinates'])

        # 在traj中添加两列数据 Lon经度, Lat纬度
        # 添加经纬度数据
        geo['Lon'] = geo['coordinates'].apply(lambda item: item[1:-1].split(',')[0])
        geo['Lat'] = geo['coordinates'].apply(lambda item: item[1:-1].split(',')[1])
        traj['Lon'] = traj['location'].map(geo['Lon'])
        traj['Lat'] = traj['location'].map(geo['Lat'])
        traj['Lon'] = traj['Lon'].astype(float)
        traj['Lat'] = traj['Lat'].astype(float)


        # print(traj.head(3))
        # print(traj['Lat'][0])
        # print(traj.dtypes)

        grid_distance = 2000
        tracks_data, grid_list = grid_process(traj, grid_distance)
        # print(tracks_data.head(3))
        # print(tracks_data.dtypes)
        # 添加地点类别信息
        traj['venue_category_id'] = traj['location'].map(geo['venue_category_id'])

        category_set = pd.unique(traj['venue_category_id'])
        id_to_index = {id_: idx for idx, id_ in enumerate(category_set)}
        tracks_data['category_index'] = tracks_data['venue_category_id'].map(id_to_index)
        # print(len(category_set))
        print(tracks_data.head(3))
        # print(tracks_data.dtypes)

        return tracks_data, grid_list


    def show_cut_data(self):
        cut_data = self.cutter_filter()

        # 展示第一个用户的轨迹
        for i, (key, value) in enumerate(cut_data.items()):
            if i >= 1:
                break
            print(key)
            # 展示每一条轨迹，value应该是轨迹的列表
            for j, item in enumerate(value):
                print(item)

    def show_encoded_data(self):
        cut_data = self.cutter_filter()   
        encoded_data = self.encode_traj(cut_data)
        data = encoded_data['encoded_data']
        for i, (key, value) in enumerate(data.items()):
            if i >= 1:
                break
            print(key)
            # 展示每一条轨迹，value应该是轨迹的列表
            for j, item in enumerate(value):
                if j >= 1:
                    break
                print(f"traj {j}")
                for k, seq in enumerate(item):
                    print(f"seq {k}")
                    print(seq)
                    print(type(seq[0]))




if __name__ == '__main__':
    from libcity.config import ConfigParser
    # 加载配置文件
    config = ConfigParser(task='traj_loc_pred', model='RNN',
                      dataset='foursquare_nyc', other_args={'batch_size': 2})
    dataset = TrajectoryTulDataset(config)
    dataset.show_cut_data()
