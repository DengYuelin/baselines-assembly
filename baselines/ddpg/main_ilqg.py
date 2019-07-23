# -*- coding: utf-8 -*-
"""
# @Time    : 08/06/19 4:47 PM
# @Author  : ZHIMIN HOU
# @FileName: iLQG.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import time

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.deepq.assembly.Env_robot_control import env_search_control

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from baselines.ddpg.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from baselines.ddpg.dynamics.dynamics_lr_prior import DynamicsLRPrior
from baselines.ddpg.dynamics.dynamics_lr import DynamicsLR

import baselines.common.tf_util as U
from baselines import logger
import numpy as np
import copy as cp

def tarin(network,
          env,
          data_path='',
          model_path='./model/',
          model_name='ddpg_none_fuzzy_150',
          file_name='test',
          model_based=False,
          memory_extend=False,
          model_type='linear',
          restore=False,
          dyna_learning=False,
          seed=None,
          nb_horizon=2,
          nb_epochs=5,   # with default settings, perform 1M steps total
          nb_sample_cycle=5,
          nb_epoch_cycles=150,
          nb_rollout_steps=400,
          nb_model_learning=10,
          nb_sample_steps=50,
          nb_samples_extend=5,
          reward_scale=1.0,
          noise_type='normal_0.2',  #'adaptive-param_0.2',  ou_0.2, normal_0.2
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50,  # per epoch cycle and MPI worker,
          batch_size=32,  # per MPI worker
          tau=0.01,
          param_noise_adaption_interval=50,
          **network_kwargs):

    for epoch in range(nb_epoch_cycles):
        for index_step in range(nb_rollout_steps):




