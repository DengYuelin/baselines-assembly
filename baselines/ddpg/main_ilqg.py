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

algorithm = {
    'iterations': 10,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-1]),
    'policy_dual_rate': 0.1,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'init_pol_wt': 0.01,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
    'max_policy_samples': 6,
    'policy_sample_mode': 'add',
}

algorithm['dynamics'] = {
        'type': DynamicsLRPrior,
        'regularization': 1e-6,
        'prior': {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 20,
        },
    }

algorithm_lr = {
        'type': DynamicsLR,
        'regularization': 1e-6,
        # 'prior': {
        #     'type': DynamicsPriorGMM,
        #     'max_clusters': 20,
        #     'min_samples_per_cluster': 40,
        #     'max_samples': 20,
        # },
    }

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

    def predict(Fm, fv, x_t, u_t, t):

        return Fm[t, :, :].dot(np.concatenate((x_t, u_t))) + fv[t, :]

    def dynamic_learning():

        """ load and process data """
        states = np.load('./train_states_noisy_ddpg_normal_0.2_epochs_5_episodes_100_none_fuzzy.npy')

        obs = np.zeros((100, 40, 12), dtype=np.float32)
        next_obs = np.zeros((100, 40, 12), dtype=np.float32)
        action = np.zeros((100, 40, 6), dtype=np.float32)
        for i in range(100):
            for j in range(40):
                obs[i, j, :] = states[0][i][j][0]
                action[i, j, :] = states[0][i][j][1]
                next_obs[i, j, :] = states[0][i][j][3]

        nn_obs = obs.reshape((4000, 12))
        nn_action = action.reshape((4000, 6))
        nn_next_obs = next_obs.reshape((4000, 12))

        """ normalize the sample data """
        vec_max = np.zeros(12)
        vec_min = np.zeros(12)
        for i in range(12):
            vec_max[i] = max(nn_obs[:, i])
            vec_min[i] = min(nn_obs[:, i])
        print('max', vec_max)
        print('min', vec_min)
        nn_obs = (nn_obs - vec_min) / (vec_max - vec_min)
        nn_next_obs = (nn_next_obs - vec_min) / (vec_max - vec_min)

        obs = nn_obs.reshape((100, 40, 12))
        next_obs = nn_next_obs.reshape((100, 40, 12))

        """ gaussian model """
        gmm_dynamic = DynamicsLRPrior(algorithm['dynamics'])

        gmm_dynamic.update_prior(obs[:100, :, :], action[:100, :, :])
        Fm_0, fv_0, dyn_covar_0 = gmm_dynamic.fit(obs[:100, :, :], action[:100, :, :])

        return


    for epoch in range(nb_epoch_cycles):
        for index_step in range(nb_rollout_steps):




