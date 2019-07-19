# -*- coding: utf-8 -*-
"""
# @Time    : 16/07/19 11:18 AM
# @Author  : ZHIMIN HOU
# @FileName: easy_example.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
from baselines.ddpg.memory import Memory
import numpy as np

memory = Memory(limit=10, action_shape=2, observation_shape=(4, ))

obs = np.zeros(5, dtype=np.float32)
action = np.zeros(2, dtype=np.float32)

# print(np.concatenate((obs, action)))
# print(memory.nb_entries)
# print(np.concatenate((memory.actions.data[:2], memory.observations0.data[:2]), axis=1))

# a = np.array([[1, 1], [2.0, 1.0], [0, 0], [0, 0], [0, 0]])
a = np.zeros(12, dtype=np.float32)
a[11] = 1
print(a[-5:])
print(a.shape)

print(slice(12))