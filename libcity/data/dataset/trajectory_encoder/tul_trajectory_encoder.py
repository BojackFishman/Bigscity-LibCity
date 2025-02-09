import os

from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'min_checkins', 'max_session_len', 'grid_size']


class TulTrajectoryEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.loc_id = 0
        self.tim_max = 47  # 时间编码方式得改变
        self.history_type = self.config['history_type']
        self.feature_dict = {'input_seq': 'int', 'time_seq': 'int',
                             'category_seq': 'int', 'input_index' : 'int', 'label': 'int'
                             }
        if config['evaluate_method'] == 'sample':
            self.feature_dict['neg_loc'] = 'int'
            parameter_list.append('neg_samples')
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))
        # 对于这种 history 模式没办法做到 batch
        if self.history_type == 'cut_off':
            # self.config['batch_size'] = 1
            self.feature_dict['history_loc'] = 'array of int'
            self.feature_dict['history_tim'] = 'array of int'

        self.grid_max = 0

    def encode(self, uid, trajectories, negative_sample=None):
        """standard encoder use the same method as DeepMove

        Recode poi id. Encode timestamp with its hour.

        Args:
            uid ([type]): same as AbstractTrajectoryEncoder
            trajectories ([type]): same as AbstractTrajectoryEncoder
                trajectory1 = [
                    (location ID, timestamp, timezone_offset_in_minutes),
                    (location ID, timestamp, timezone_offset_in_minutes),
                    .....
                ]
        
        encode的输入数据是一个用户的所有轨迹，是轨迹的列表
        encode的输出数据同样是一个列表，是编码后的一个用户的轨迹；每一条轨迹的轨迹点聚合为了五个列表，由轨迹点的横切变为列表的竖切
            encode_trajectories = [
                trajectory1 = [
                    input_seq,
                    time_seq,
                    category_seq,
                    input_index,
                    label
                ]
                trajectory2 = [
                    ...
                ]
            ]
        """
        # 直接对 uid 进行重编码
        uid = self.uid
        self.uid += 1
        encoded_trajectories = []
        # history_loc = []
        # history_tim = []

        # todo 处理网格，将轨迹点的位置转换为网格id


        for index, traj in enumerate(trajectories):
            input_seq = []
            time_seq = []
            category_seq = []
            input_index = []
            label = []

            input_index.append(traj[0][10])  # 轨迹编号
            label.append(uid)
            # 获取轨迹点id 
            # 获取时间信息
            # 获取轨迹点的类别信息
            # traj_point_ids = []
            # traj_point_times = []
            # traj_point_category_ids = []
            for point in traj:
                # loc = point[4]
                grid = point[7]
                input_seq.append(grid)
                self.grid_max = max(self.grid_max, grid)  # 求grid的最大值，+1用于pad值
          
                now_time = parse_time(point[2])
                # 采用工作日编码到0-23，休息日编码到24-47
                time_code = self._time_encode(now_time)
                time_seq.append(time_code)      

                category_seq.append(point[9])

                # if loc not in self.location2id:
                #     self.location2id[loc] = self.loc_id
                #     self.loc_id += 1
            
            # input_seq.append(traj_point_ids)
            # time_seq.append(traj_point_times)
            # category_seq.append(traj_point_category_ids)

            trace = []
            trace.append(input_seq)
            trace.append(time_seq)
            trace.append(category_seq)
            trace.append(input_index)
            trace.append(label)

            encoded_trajectories.append(trace)

        return encoded_trajectories

    def gen_data_feature(self):
        """
        encode的输出数据同样是一个列表，是编码后的一个用户的轨迹；每一条轨迹的轨迹点聚合为了五个列表，由轨迹点的横切变为列表的竖切
            encode_trajectories = [
                trajectory1 = [
                    input_seq,
                    time_seq,
                    category_seq,
                    input_index,
                    label
                ]
                trajectory2 = [
                    ...
                ]
            ]
        """
        grid_pad = self.grid_max + 1
        tim_pad = self.tim_max + 1        
        category_pad = 398
        self.pad_item = {
            'input_seq': grid_pad,
            'time_seq': tim_pad,
            'category_seq': category_pad,
        }

        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'grid_pad': grid_pad,
            'tim_pad': tim_pad,
            'category_pad': category_pad
        }

    def _time_encode(self, time):
        if time.weekday() in [0, 1, 2, 3, 4]:
            return time.hour
        else:
            return time.hour + 24
