# -*- coding: utf-8 -*-
"""
# @Time    : 23/07/19 8:51 PM
# @Author  : ZHIMIN HOU
# @FileName: hyperparams.py.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ABB.agent_abb import AgentABB
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr import DynamicsLR
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import tf_network


PR2_GAINS = np.array([1., 1., 1., 1., 1., 1.])

SENSOR_DIMS = {
    'POS_FORCE': 12,
    'ACTION': 6
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/assembly_robotic_example/'

common = {
    'experiment_name': 'assembly_robotic_example' + '_' +\
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentABB,
    'target_state': np.array([-0, -0, -0, -0, -0, 0, 1453.2509, 73.2577, 980, 179.8938, 0.9185, 1.0311]),  # target state
    'render': False,
    'x0': np.array([-0, -0, -0, -0, -0, 0, 1453.2509, 73.2577, 995.8843, 179.8938, 0.9185, 1.0311]),
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': ['POS_FORCE'],
    'obs_include': ['POS_FORCE'],
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS['ACTION']),
    'init_acc': np.zeros(SENSOR_DIMS['ACTION']),
    'init_var': 0.1,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm.update({
    'sample_on_policy': True,
})

torque_cost = {
    'type': CostAction,
    'wu': 1e-3 / PR2_GAINS,
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.array([2, 2, 1, 2, 2, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

# Create second cost function for last step only
final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,
    'target_end_effector': fk_cost['target_end_effector'],
    'wp': fk_cost['wp'],
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 1e-5,
    'wp_final_multiplier': 10.0,
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1., 1., 1., 1., 1., 1.])
}

state_cost = {
    'type': CostState,
    'data_types': {
        'POS_FORCE': {
            'wp': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-5, 1.0],
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

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': ['POS_FORCE'],
        'obs_vector_data': ['POS_FORCE'],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': tf_network,
    'iterations': 3000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'gui_on': False,
    'iterations': 100,
    'num_samples': 10,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'smooth_noise_var': 0.2,
    'smooth_noise': True
}

common['info'] = generate_experiment_info(config)
