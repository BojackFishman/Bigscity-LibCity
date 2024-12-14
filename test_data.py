# import sys
# import os

# root_path = os.path.abspath(__file__)
# root_path = '\\'.join(root_path.split('\\')[:-2])
# sys.path.append(root_path)
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model, get_executor, get_logger, set_random_seed
import random
from libcity.data.dataset.trajectory_tul_dataset import TrajectoryTulDataset

"""
取一个batch的数据进行初步测试
Take the data of a batch for preliminary testing
"""

# 加载配置文件
config = ConfigParser(task='traj_tul', model='Attn',
                      dataset='foursquare_nyc', other_args={'batch_size': 2})
# config = ConfigParser(task='traj_loc_pred', model='RNN',
#                       dataset='foursquare_nyc', other_args={'batch_size': 2})
exp_id = config.get('exp_id', None)
if exp_id is None:
    exp_id = int(random.SystemRandom().random() * 100000)
    config['exp_id'] = exp_id
# logger
logger = get_logger(config)
logger.info(config.config)
# seed
seed = config.get('seed', 0)
set_random_seed(seed)

# 验证编码前的数据
# dataset = TrajectoryTulDataset(config)
# dataset.gen_traj()
# # dataset.show_cut_data()
# # dataset.show_encoded_data()

# 验证编码后并且通过dataloader的数据
dataset = get_dataset(config)
# dataset.gen_tarj()

# 数据预处理，划分数据集
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
# 抽取一个 batch 的数据进行测试
batch = train_data.__iter__().__next__()
batch.to_tensor(device=config['device'])
print(batch.data)

# 加载模型
model = get_model(config, data_feature)
model = model.to(config['device'])
# 加载执行器
executor = get_executor(config, model, data_feature)
print("model and executor loaded OK.")

# 查看batch中每一个数据的类型
# for key in batch.data:
#     print(f"Feature: {key}, Data Type: {type(batch.data[key])}")
#     print(batch.data[key])
#     types = []
#     for item in batch.data[key]:
#         for val in item:
#             types.append(type(val))
#     print(types)


# 加载数据模块
# dataset = get_dataset(config)
# # 数据预处理，划分数据集
# train_data, valid_data, test_data = dataset.get_data()

# print("hello world")
# for i, batch in enumerate(train_data):
#     if i >= 10:
#         break
#     print(f"Batch {i+1}: {batch}")

# batch = train_data.__iter__().__next__()
# print(f"Attributes: {dir(batch)}")
# print(f"Data: {vars(batch)}")

# for i in range(10) :
#     print(train_data.dataset[i])




# data_feature = dataset.get_data_feature()
# # 抽取一个 batch 的数据进行模型测试
# batch = train_data.__iter__().__next__()
# # 加载模型
# model = get_model(config, data_feature)
# model = model.to(config['device'])
# # 加载执行器
# executor = get_executor(config, model, data_feature)
# # 模型预测
# batch.to_tensor(config['device'])
# res = model.predict(batch)
# logger.info('Result shape is {}'.format(res.shape))
# logger.info('Success test the model!')
