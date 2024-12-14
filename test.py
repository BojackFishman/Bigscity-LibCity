print("hello world")

import sys
import os
import json

# root_path = os.path.abspath(__file__)
# print(root_path)
# root_path = '\\'.join(root_path.split('\\')[:-2])
# print(root_path)

# print(os.path.abspath(__file__))
cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory__gowalla_5_2_StandardTrajectoryEncoder_time_interval_72_splice_3_10.json')
f = open(cache_file_name, 'r')
data = json.load(f)