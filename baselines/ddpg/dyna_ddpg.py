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

import baselines.common.tf_util as U
from baselines import logger
import numpy as np
import copy as cp


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

    if model_based:
        """ store fake_data"""
        fake_memory = Memory(limit=int(1e5), action_shape=env.action_space.shape[0], observation_shape=env.observation_space.shape)

        """ select model or not """
        if model_type == 'gp':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            dynamic_model = GaussianProcessRegressor(kernel=kernel)
            reward_model = GaussianProcessRegressor(kernel=kernel)
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
                        m_new_obs = dynamic_model.predict(pred_x)[0]
                        """ get real reward """
                        # state = env.inverse_state(m_new_obs)
                        # m_reward = env.get_reward(state, m_action)
                        m_reward = reward_model.predict(pred_x)[0]
                        agent.store_transition(obs, m_action, m_reward, m_new_obs, done)

            if cycle > nb_model_learning and cycle//5 == 0:

                dynamic_index = dynamic_episode_steps[-5:cycle]
                length = min(dynamic_index)
                dynamic_X = dynamic_states[-5:cycle, :length, :]
                dynamic_U = dynamic_actions[-5:cycle, :length, :]




            """ generate new data and fit model"""
            if model_based and cycle > nb_model_learning:
                logger.info("==============================  Model Fit !!! ===============================")
                input_x = np.concatenate((memory.observations0.data[:memory.nb_entries], memory.actions.data[:memory.nb_entries]), axis=1)
                input_y_obs = memory.observations1.data[:memory.nb_entries]
                input_y_reward = memory.rewards.data[:memory.nb_entries]
                dynamic_model.fit(input_x, input_y_obs)
                reward_model.fit(input_x, input_y_reward)

                if dyna_learning:
                    logger.info("=========================  Collect data !!! =================================")
                    pred_obs = np.zeros((1, 18), dtype=np.float32)
                    for sample_index in range(nb_sample_cycle):
                        fake_obs = obs_reset
                        for t_episode in range(nb_sample_steps):
                            fake_action, _, _, _ = agent.step(fake_obs, stddev, apply_noise=True, compute_Q=False)
                            pred_obs[:, :12] = fake_obs
                            pred_obs[:, 12:] = fake_action
                            next_fake_obs = dynamic_model.predict(pred_obs)[0]
                            fake_reward = reward_model.predict(pred_obs)[0]
                            # next_fake_obs = dynamic_model.predict(np.concatenate((fake_obs, fake_action)))[0]
                            # fake_reward = reward_model.predict(np.concatenate((fake_obs, fake_action)))[0]
                            fake_obs = next_fake_obs
                            fake_terminals = False
                            fake_memory.append(fake_obs, fake_action, fake_reward, next_fake_obs, fake_terminals)

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

    algorithm_name = 'noisy_ddpg'
    env = env_search_control(step_max=200, fuzzy=True, add_noise=True)
    data_path = './prediction_data/'
    model_path = './prediction_model/'
    file_name = '_epochs_5_episodes_100_fuzzy'
    model_name = './prediction_model/'
    learn(network='mlp',
          env=env,
          data_path=data_path,
          model_based=False,
          memory_extend=False,
          dyna_learning=False,
          model_type='gp',
          noise_type='normal_0.2',
          file_name=file_name,
          model_path=model_path,
          model_name=model_name,
          restore=False,
          nb_epochs=5,
          nb_sample_steps=50,
          nb_samples_extend=10,
          nb_model_learning=30,
          nb_epoch_cycles=100,
          nb_train_steps=60,
          nb_rollout_steps=200)

    """
    Model-based gaussian model
    Episode_Rewards:: [-6.559999999999991, -6.244999999999991, -5.824999999999993, -5.614999999999993, -5.824999999999993, -6.979999999999989, -7.609999999999987, -8.239999999999984, -8.344999999999985, -8.449999999999983, -8.449999999999983, -8.449999999999983, -8.554999999999984, -8.449999999999983, -8.449999999999983, -8.449999999999983, -8.239999999999984, -7.189999999999988, -5.7199999999999935, -5.5099999999999945, -5.5099999999999945, -5.404999999999994, -5.404999999999994, -5.5099999999999945, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.404999999999994, -5.5099999999999945, -5.5099999999999945, -5.5099999999999945, -5.5099999999999945, -5.5099999999999945, -5.404999999999994, -5.404999999999994, -5.5099999999999945, -5.5099999999999945, -5.5099999999999945, -5.5099999999999945, -5.5099999999999945]
    Episode_Times:: [27.767322540283203, 29.839134454727173, 28.626702785491943, 28.169037342071533, 28.601523399353027, 31.77764320373535, 33.55309987068176, 35.18787622451782, 35.532920598983765, 35.774473667144775, 35.79237246513367, 35.7697548866272, 36.09323000907898, 35.901691198349, 35.81383466720581, 35.72635340690613, 35.08375811576843, 32.345478534698486, 28.381656646728516, 27.768736124038696, 27.707700490951538, 27.490614652633667, 27.4447762966156, 27.67473077774048, 27.48730778694153, 27.45588207244873, 27.37090492248535, 27.31996488571167, 30.307120323181152, 27.780642986297607, 27.495296955108643, 237.87087750434875, 225.64937281608582, 317.86939120292664, 375.10274052619934, 354.0145788192749, 338.8398802280426, 278.26734805107117, 278.65738344192505, 296.736270904541, 312.2680208683014, 441.1866557598114, 343.715788602829, 647.8456799983978, 371.28341841697693, 406.30851197242737]
    Episode_Rewards:: [-6.76999999999999, -6.559999999999991, -6.139999999999992, -5.824999999999993, -6.034999999999992, -6.979999999999989, -7.819999999999986, -8.029999999999985, -7.924999999999986, -7.924999999999986, -7.819999999999986, -7.609999999999987, -7.189999999999988, -7.7149999999999865, -8.449999999999983, -8.554999999999984, -8.764999999999983, -8.869999999999983, -8.869999999999983, -8.764999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.974999999999982, -8.869999999999983, -8.764999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.764999999999983, -8.659999999999982, -8.659999999999982, -8.869999999999983, -8.869999999999983, -8.974999999999982, -8.869999999999983, -8.764999999999983, -8.764999999999983, -8.764999999999983, -8.869999999999983, -8.869999999999983, -8.869999999999983, -8.974999999999982, -8.869999999999983, -8.869999999999983, -8.974999999999982, -8.974999999999982, -8.974999999999982, -8.974999999999982, -8.974999999999982, -8.974999999999982, -8.974999999999982]
    Episode_Times:: [26.994033813476562, 30.770112991333008, 29.46457576751709, 28.724473237991333, 29.20805025100708, 31.792637825012207, 34.08474898338318, 34.591211795806885, 34.31247019767761, 34.40872144699097, 34.07383894920349, 33.48011112213135, 32.39230966567993, 33.83802938461304, 35.78772282600403, 35.97193002700806, 36.62266278266907, 36.860942125320435, 36.82782769203186, 36.52959322929382, 36.97553730010986, 36.84327554702759, 36.81550455093384, 36.933107137680054, 36.72500038146973, 36.80849766731262, 36.78366255760193, 37.31751322746277, 36.87189316749573, 36.66956615447998, 36.825047969818115, 319.5994963645935, 371.5126769542694, 390.478036403656, 473.81298327445984, 554.5190415382385, 423.5883288383484, 480.93056631088257, 456.9679927825928, 608.2720158100128, 519.0065865516663, 517.6432852745056, 678.5634579658508, 1131.3627626895905, 830.992062330246, 671.4969174861908, 740.5868232250214, 816.2626433372498, 1049.6738183498383, 1089.3690526485443, 963.5303378105164, 1123.8543672561646, 1213.2957389354706, 1147.394606590271, 1162.4059371948242, 1003.5584232807159, 1636.1836168766022, 1327.16890001297]
    """