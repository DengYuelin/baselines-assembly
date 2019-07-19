# -*- coding: utf-8 -*-
"""
# @Time    : 18/07/19 3:56 PM
# @Author  : ZHIMIN HOU
# @FileName: dyna_ddpg.py
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

    return Fm[t, :, :].dot(np.concatenate(x_t, u_t)) + fv[t, :]


def learn(network,
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

    nb_actions = env.action_space.shape[0]
    memory = Memory(limit=int(1e5), action_shape=env.action_space.shape[0], observation_shape=env.observation_space.shape)

    Fm = None
    fv = None
    dyn_covar =None
    length = 100
    if model_based:
        """ store fake_data"""
        fake_memory = Memory(limit=int(1e5), action_shape=env.action_space.shape[0], observation_shape=env.observation_space.shape)

        """ select model or not """
        if model_type == 'gp':
            # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            # dynamic_model = GaussianProcessRegressor(kernel=kernel)
            # reward_model = GaussianProcessRegressor(kernel=kernel)
            lr_dynamic = DynamicsLRPrior(algorithm['dynamics'])
            # gmm_dynamic = DynamicsPriorGMM(algorithm['dynamics'])
        elif model_type == 'linear':
            dynamic_model = LinearRegression()
            reward_model = LinearRegression()
        elif model_type == 'mlp':
            dynamic_model = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                                         alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                                         max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                                         nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-08)
            reward_model = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                                         alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                                         max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                                         nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-08)
        else:
            logger.info("You need to give the model_type to fit the dynamic and reward!!!")

    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    """ set noise """
    action_noise = None
    param_noise = None

    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    """action scale"""
    max_action = env.action_high_bound
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    """ agent ddpg """
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape[0],
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm, reward_scale=reward_scale)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    sess = U.get_session()
    if restore:
        agent.restore(sess, model_path, model_name)
    else:
        agent.initialize(sess)
        sess.graph.finalize()

    agent.reset()
    episodes = 0
    epochs_rewards = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_times = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_steps = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_states = []
    dynamic_states = np.zeros((nb_epoch_cycles, nb_rollout_steps, env.observation_dim), dtype=np.float32)
    dynamic_actions = np.zeros((nb_epoch_cycles, nb_rollout_steps, env.action_dim), dtype=np.float32)
    dynamic_episode_steps = np.zeros(nb_epoch_cycles, dtype=np.float32)

    for epoch in range(nb_epochs):
        logger.info("======================== The {} epoch start !!! =========================".format(epoch))
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_times = []
        epoch_actions = []
        epoch_episode_states = []
        epoch_qs = []
        epoch_episodes = 0
        for cycle in range(nb_epoch_cycles):
            start_time = time.time()
            obs, state, done = env.reset()
            obs_reset = cp.deepcopy(obs)
            episode_reward = 0.
            episode_step = 0
            episode_states = []
            logger.info("================== The {} episode start !!! ===================".format(cycle))
            for t_rollout in range(nb_rollout_steps):
                logger.info("================== The {} steps finish  !!! ===================".format(t_rollout))

                """ Predict next action """
                action, q, _, _ = agent.step(obs, stddev, apply_noise=True, compute_Q=True)

                """ store dynamic data """
                dynamic_actions[cycle, t_rollout, :] = cp.deepcopy(action)
                dynamic_states[cycle, t_rollout, :] = cp.deepcopy(obs)
                new_obs, next_state, r, done, safe_or_not, final_action = env.step(max_action * action, t_rollout)

                if safe_or_not is False:
                    break

                episode_reward += r
                episode_step += 1
                episode_states.append([cp.deepcopy(state), cp.deepcopy(final_action), np.array(cp.deepcopy(r)), cp.deepcopy(next_state)])

                epoch_actions.append(action)
                epoch_qs.append(q)

                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs
                state = next_state

                if done:
                    break

                """ extend the memory """
                if model_based and cycle > (nb_model_learning + 1) and memory_extend:
                    pred_x = np.zeros((1, 18), dtype=np.float32)
                    for j in range(nb_samples_extend):
                        m_action, _, _, _ = agent.step(obs, stddev, apply_noise=True, compute_Q=False)
                        pred_x[:, :12] = obs
                        pred_x[:, 12:] = m_action
                        # m_new_obs = dynamic_model.predict(pred_x)[0]
                        if t_rollout < length:
                            m_new_obs = predict(Fm, fv, obs, action, t_rollout)
                            """ get real reward """
                            state = env.inverse_state(m_new_obs)
                            m_reward = env.get_reward(state, m_action)
                            # m_reward = reward_model.predict(pred_x)[0]
                            agent.store_transition(obs, m_action, m_reward, m_new_obs, done)

            """ generate new data and fit model"""
            if cycle > nb_model_learning and cycle//nb_horizon == 0:

                dynamic_index = dynamic_episode_steps[-nb_horizon:cycle]
                length = min(dynamic_index)
                dynamic_X = dynamic_states[-nb_horizon:cycle, :length, :]
                dynamic_U = dynamic_actions[-nb_horizon:cycle, :length, :]

                lr_dynamic.update_prior(dynamic_X, dynamic_U)
                Fm, fv, dyn_covar = lr_dynamic.fit(dynamic_X, dynamic_U)

            # """ generate new data and fit model"""
            # if model_based and cycle > nb_model_learning:
            #     logger.info("==============================  Model Fit !!! ===============================")
            #     input_x = np.concatenate((memory.observations0.data[:memory.nb_entries], memory.actions.data[:memory.nb_entries]), axis=1)
            #     input_y_obs = memory.observations1.data[:memory.nb_entries]
            #     input_y_reward = memory.rewards.data[:memory.nb_entries]
            #     dynamic_model.fit(input_x, input_y_obs)
            #     reward_model.fit(input_x, input_y_reward)
            #
            #     if dyna_learning:
            #         logger.info("=========================  Collect data !!! =================================")
            #         pred_obs = np.zeros((1, 18), dtype=np.float32)
            #         for sample_index in range(nb_sample_cycle):
            #             fake_obs = obs_reset
            #             for t_episode in range(nb_sample_steps):
            #                 fake_action, _, _, _ = agent.step(fake_obs, stddev, apply_noise=True, compute_Q=False)
            #                 pred_obs[:, :12] = fake_obs
            #                 pred_obs[:, 12:] = fake_action
            #                 next_fake_obs = dynamic_model.predict(pred_obs)[0]
            #                 fake_reward = reward_model.predict(pred_obs)[0]
            #                 # next_fake_obs = dynamic_model.predict(np.concatenate((fake_obs, fake_action)))[0]
            #                 # fake_reward = reward_model.predict(np.concatenate((fake_obs, fake_action)))[0]
            #                 fake_obs = next_fake_obs
            #                 fake_terminals = False
            #                 fake_memory.append(fake_obs, fake_action, fake_reward, next_fake_obs, fake_terminals)

            """ noise decay """
            stddev = float(stddev) * 0.95

            duration = time.time() - start_time
            epoch_episode_rewards.append(episode_reward)
            epoch_episode_steps.append(episode_step)
            epoch_episode_times.append(cp.deepcopy(duration))
            epoch_episode_states.append(cp.deepcopy(episode_states))

            epochs_rewards[epoch, cycle] = episode_reward
            epochs_steps[epoch, cycle] = episode_step
            epochs_times[epoch, cycle] = cp.deepcopy(duration)

            logger.info("============================= The Episode_Times:: {}!!! ============================".format(epoch_episode_rewards))
            logger.info("============================= The Episode_Times:: {}!!! ============================".format(epoch_episode_times))

            epoch_episodes += 1
            episodes += 1

            """ Training process """
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                logger.info("")
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)
                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            """ planning training """
            if model_based and cycle > (nb_model_learning + 1) and dyna_learning:
                for t_train in range(nb_train_steps):
                    # setting for adapt param noise, if necessary.
                    if fake_memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)
                    batch = fake_memory.sample(batch_size=batch_size)
                    fake_cl, fake_al = agent.train_fake_data(batch)
                    epoch_critic_losses.append(fake_cl)
                    epoch_actor_losses.append(fake_al)
                    agent.update_target_net()

        epochs_states.append(cp.deepcopy(epoch_episode_states))

        # # save data
        np.save(data_path + 'train_reward_' + algorithm_name + '_' + noise_type + file_name, epochs_rewards)
        np.save(data_path + 'train_step_' + algorithm_name + '_' + noise_type + file_name, epochs_steps)
        np.save(data_path + 'train_states_' + algorithm_name + '_' + noise_type + file_name, epochs_states)
        np.save(data_path + 'train_times_' + algorithm_name + '_' + noise_type + file_name, epochs_times)

    # # agent save
    agent.store(model_path + 'train_model_' + algorithm_name + '_' + noise_type + file_name)


if __name__ == '__main__':

    algorithm_name = 'noisy_extend_gp_ddpg'
    env = env_search_control(step_max=200, fuzzy=False, add_noise=False)
    data_path = './prediction_data/'
    model_path = './prediction_model/'
    file_name = '_epochs_5_episodes_100_fuzzy'
    model_name = './prediction_model/'

    learn(network='mlp',
          env=env,
          data_path=data_path,
          model_based=True,
          memory_extend=True,
          dyna_learning=False,
          model_type='gp',
          noise_type='normal_0.2',
          file_name=file_name,
          model_path=model_path,
          model_name=model_name,
          restore=False,
          nb_horizon=5,
          nb_epochs=5,
          nb_sample_steps=50,
          nb_samples_extend=10,
          nb_model_learning=4,
          nb_epoch_cycles=100,
          nb_train_steps=60,
          nb_rollout_steps=200)

    # states = np.load('./prediction_data/train_states_dyna_nn_ddpg_normal_0.2_epochs_5_episodes_100_none_fuzzy.npy')
    #
    # obs = np.zeros((5, 40, 12), dtype=np.float32)
    # action = np.zeros((5, 40, 6), dtype=np.float32)
    # for i in range(5):
    #     for j in range(40):
    #         obs[i, j, :] = states[0][i][j][0]
    #         action[i, j, :] = states[0][i][j][1]
    #
    #
    # lr_dynamic.update_prior(obs, action)
    # Fm, fv, dyn_covar = lr_dynamic.fit(obs, action)
    # sigma = np.zeros((40, 18, 18), dtype=np.float32)
    # mu = np.zeros((40, 18), dtype=np.float32)

    # id_x =slice(12)
    # sigma[1, id_x, id_x] = Fm[1, :, :].dot(sigma[1, :, :]).dot(Fm[1, :, :].T) + dyn_covar[1, :, :]
    # # mu[t + 1, id_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
    # print(sigma)

    # print(obs[0][0][:])
    # print(action[0][0][:])
    # print((Fm[0, :, :].dot(np.concatenate([obs[0][0][:], action[0][0][:]]))))
    # print(V.shape)

    # if t < T - 1:
    #     sigma[t + 1, idx_x, idx_x] = \
    #         Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
    #         dyn_covar[t, :, :]
    #     mu[t + 1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

    # prior = lr_dynamic.get_prior()

    # # Fit x0mu/x0sigma.
    # x0 = X[:, 0, :]
    # x0mu = np.mean(x0, axis=0)
    # self.cur[m].traj_info.x0mu = x0mu
    # self.cur[m].traj_info.x0sigma = np.diag(
    #     np.maximum(np.var(x0, axis=0),
    #                self._hyperparams['initial_state_var'])
    # )
    #
    # prior = self.cur[m].traj_info.dynamics.get_prior()
    # if prior:
    #     mu0, Phi, priorm, n0 = prior.initial_state()
    #     N = len(cur_data)
    #     self.cur[m].traj_info.x0sigma += \
    #         Phi + (N * priorm) / (N + priorm) * \
    #         np.outer(x0mu - mu0, x0mu - mu0) / (N + n0)
