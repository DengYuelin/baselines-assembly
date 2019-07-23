# -*- coding: utf-8 -*-
"""
# @Time    : 18/07/19 3:56 PM
# @Author  : ZHIMIN HOU
# @FileName: dyna_ddpg.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from baselines.ddpg.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from baselines.ddpg.dynamics.dynamics_lr_prior import DynamicsLRPrior
from baselines.ddpg.dynamics.dynamics_lr import DynamicsLR
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


def calculate_dynamic(Fm, fv, dyn_covar):
    T = Fm.shape[0]
    dX = Fm.shape[1]
    X_U = Fm.shape[2]

    # Allocate space.
    sigma = np.zeros((T, X_U, X_U))
    mu = np.zeros((T, X_U))

    id_x = slice(dX)
    for t in range(T):
        sigma[t + 1, id_x, id_x] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + dyn_covar[t, :, :]
        mu[t + 1, id_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

    return mu, sigma


def predict(Fm, fv, x_t, u_t, t):

    return Fm[t, :, :].dot(np.concatenate((x_t, u_t))) + fv[t, :]


if __name__ == "__main__":

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
    nn_next_obs =next_obs.reshape((4000, 12))

    """ normalize the sample data """
    vec_max = np.zeros(12)
    vec_min = np.zeros(12)
    for i in range(12):
        vec_max[i] = max(nn_obs[:, i])
        vec_min[i] = min(nn_obs[:, i])
    print('max', vec_max)
    print('min', vec_min)
    nn_obs = (nn_obs - vec_min)/(vec_max - vec_min)
    nn_next_obs = (nn_next_obs - vec_min)/(vec_max - vec_min)

    obs = nn_obs.reshape((100, 40, 12))
    next_obs = nn_next_obs.reshape((100, 40, 12))

    """ ==================== fit the dynamic model ======================== """
    """ gaussian model """
    gmm_dynamic = DynamicsLRPrior(algorithm['dynamics'])

    """ linear model """
    lr_dynamic = DynamicsLR(algorithm_lr)

    """ non-linear model """
    mlp_dynamic = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                 alpha=0.0001, batch_size='auto', learning_rate='constant',
                                 learning_rate_init=0.001, power_t=0.5,
                                 max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
                                 warm_start=False, momentum=0.9,
                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08)
    # linear_dynamic = LinearRegression()

    # dynamics = {'dynamic_0': gmm_dynamic, 'dynamic_1': lr_dynamic, 'dynamic_2': mlp_dynamic}
    # print(dynamics['dynamic_0'])

    gmm_dynamic.update_prior(obs[:90, :, :], action[:90, :, :])
    Fm_0, fv_0, dyn_covar_0 = gmm_dynamic.fit(obs[:90, :, :], action[:90, :, :])

    Fm_1, fv_1, dyn_covar_1 = lr_dynamic.fit(obs[:90, :, :], action[:90, :, :])

    """ scikit-learn regression """
    mlp_dynamic.fit(np.concatenate((nn_obs[:3900, :], nn_action[:3900, :]), axis=1), nn_next_obs[:3900, :])
    # linear_dynamic.fit(np.concatenate((nn_obs, nn_action), axis=1), nn_next_obs)

    """ evaluate predict error """
    # predit_error_lr = np.zeros((), dtype=np.float32)
    # predit_error_gmm = np.zeros((), dtype=np.float32)
    # predit_error_nn = np.zeros((), dtype=np.float32)
    # eval_states = np.load('./train_states_noisy_ddpg_normal_0.2_epochs_5_episodes_100_fuzzy.npy')
    # eval_obs = np.zeros((100, 40, 12), dtype=np.float32)
    # eval_next_obs = np.zeros((100, 40, 12), dtype=np.float32)
    # eval_action = np.zeros((100, 40, 6), dtype=np.float32)
    # for i in range(100):
    #     for j in range(40):
    #         eval_obs[i, j, :] = eval_states[0][i][j][0]
    #         eval_action[i, j, :] = eval_states[0][i][j][1]
    #         eval_next_obs[i, j, :] = eval_states[0][i][j][3]

    predict_error_lr = np.zeros((10, 40), dtype=np.float32)
    predict_error_gmm = np.zeros((10, 40), dtype=np.float32)
    predict_error_nn = np.zeros((10, 40), dtype=np.float32)
    for i in range(10):
        for index_t in range(40):
            # print(np.array(obs[i, index_t, :]))
            # print(np.concatenate(np.array(obs[i, index_t, :]), np.array(action[i, index_t, :])))
            # predict_next_obs = predict(Fm_1, fv_1, obs[i, index_t, :], action[i, index_t, :], index_t)
            # print((obs[i, index_t, :] - vec_min) / (vec_max - vec_min))
            predict_next_obs_gmm = predict(Fm_0, fv_0, obs[99-i, index_t, :], action[99-i, index_t, :], index_t)
            predict_next_obs_lr = predict(Fm_1, fv_1, obs[99-i, index_t, :], action[99-i, index_t, :], index_t)
            predict_next_obs_nn = mlp_dynamic.predict(np.concatenate((obs[99-i, index_t, :], action[99-i, index_t, :])).reshape(-1, 18))
            predict_error_lr[i, index_t] = np.mean(predict_next_obs_lr - next_obs[99-i, index_t, :])
            predict_error_gmm[i, index_t] = np.mean(predict_next_obs_gmm - next_obs[99-i, index_t, :])
            predict_error_nn[i, index_t] = np.mean(predict_next_obs_nn - next_obs[99-i, index_t, :])
    final_error_lr = np.mean(predict_error_lr)
    std_error_lr = np.std(predict_error_lr)
    final_error_gmm = np.mean(predict_error_gmm)
    std_error_gmm = np.std(predict_error_gmm)
    final_error_nn = np.mean(predict_error_nn)
    std_error_nn = np.std(predict_error_nn)
    print('final_error_lr', final_error_lr)
    print('std_error_lr', std_error_lr)
    print('final_error_gmm', final_error_gmm)
    print('std_error_gmm', std_error_gmm)
    print('final_error_nn', final_error_nn)
    print('std_error_nn', std_error_nn)
