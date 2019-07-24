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

from iLQG import iLQR, fd_Cost, fd_Dynamics, myCost

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


def train(network,
          env,
          data_path='',
          model_path='./model/',
          model_name='ddpg_none_fuzzy_150',
          algorithm_name='',
          file_name='test',
          model_based=False,
          memory_extend=False,
          model_type='linear',
          restore=False,
          dyna_learning=False,
          seed=None,
          nb_epochs=5, # with default settings, perform 1M steps total
          nb_sample_cycle=5,
          nb_epoch_cycles=150,
          nb_rollout_steps=400,
          nb_model_learning=10,
          nb_sample_steps=50,
          nb_samples_extend=5,
          reward_scale=1.0,
          noise_type='normal_0.2', #'adaptive-param_0.2',  ou_0.2, normal_0.2
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
    memory = Memory(limit=int(1e5), action_shape=env.action_space.shape[0],
                    observation_shape=env.observation_space.shape)

    if model_based:
        """ store fake_data"""
        fake_memory = Memory(limit=int(1e5), action_shape=env.action_space.shape[0], observation_shape=env.observation_space.shape)

        """ select model or not """
        if model_type == 'gp':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            dynamic_model = GaussianProcessRegressor(kernel=kernel)
            reward_model = GaussianProcessRegressor(kernel=kernel)

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

                return Fm_0, fv_0, dyn_covar_0

        elif model_type == 'linear':
            dynamic_model = LinearRegression()
            # reward_model = LinearRegression()
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

    def f(x, u):
        assert len(x) == env.observation_dim, x.shape
        assert len(u) == env.observation_dim, u.shape
        x_ = dynamic_model.predict(x, u)
        return x_

    def l(x, u):
        reward = env.get_running_cost(x, u)
        return reward

    def l_terminal(x):
        reward = env.get_reward_terminal(x)
        return reward

    dynamics = fd_Dynamics(f, env.observation_dim, env.action_dim)
    cost = fd_Cost(l, l_terminal, env.observation_dim, env.action_dim)

    episodes = 0
    epochs_rewards = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_times = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_steps = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_states = []
    ilqr_a_init = np.zeros(env.action_dim)

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
                action_ddpg, q, _, _ = agent.step(obs, stddev, apply_noise=True, compute_Q=True)

                """ action derived from ilqg """
                if cycle > (nb_model_learning + 1):
                    # Number of time-steps in trajectory.
                    N = 1
                    # Initial state.
                    x_init = obs
                    # Random initial action path.
                    u_init = np.array([ilqr_a_init])

                    ilqr = iLQR(dynamics, cost, N)
                    xs, us = ilqr.fit(x_init, u_init)

                    a_raw = us[0]
                    a_raw[1] = -abs(a_raw[1])
                    a_raw = np.tanh(a_raw)

                    noise = action_noise(stddev)
                    action = a_raw + noise

                else:
                    action = action_ddpg

                """ Execute the derived action """
                new_obs, next_state, r, done, safe_or_not, final_action = env.step(max_action * action, t_rollout)

                if safe_or_not is False:
                    break

                episode_reward += r
                episode_step += 1
                episode_states.append(
                    [cp.deepcopy(state), cp.deepcopy(final_action), np.array(cp.deepcopy(r)), cp.deepcopy(next_state)])

                epoch_actions.append(action)
                epoch_qs.append(q)

                agent.store_transition(obs, action, r, new_obs, done)

                ilqr_a_init = action_ddpg.copy()
                obs = new_obs
                state = next_state

                if done:
                    break

            """ generate new data and fit model"""
            if model_based and cycle > nb_model_learning:
                logger.info("==============================  Model Fit !!! ===============================")
                input_x = np.concatenate(
                    (memory.observations0.data[:memory.nb_entries], memory.actions.data[:memory.nb_entries]), axis=1)
                input_y_obs = memory.observations1.data[:memory.nb_entries]
                input_y_reward = memory.rewards.data[:memory.nb_entries]
                dynamic_model.fit(input_x, input_y_obs)
                # reward_model.fit(input_x, input_y_reward)

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

            logger.info("============================= The Episode_Reward:: {}!!! ============================".format(
                epoch_episode_rewards))
            logger.info("============================= The Episode_Times:: {}!!! ============================".format(
                epoch_episode_times))

            epoch_episodes += 1
            episodes += 1

            """ Training process """
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                logger.info(" ============================ DDPG-Training =========================")
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)
                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()


        epochs_states.append(cp.deepcopy(epoch_episode_states))

        # save data
        np.save(data_path + 'train_reward_' + algorithm_name + '_' + noise_type + file_name, epochs_rewards)
        np.save(data_path + 'train_step_' + algorithm_name + '_' + noise_type + file_name, epochs_steps)
        np.save(data_path + 'train_states_' + algorithm_name + '_' + noise_type + file_name, epochs_states)
        np.save(data_path + 'train_times_' + algorithm_name + '_' + noise_type + file_name, epochs_times)

    # agent save
    agent.store(model_path + 'train_model_' + algorithm_name + '_' + noise_type + file_name)


if __name__ == '__main__':

    algorithm_name = 'noisy_ilqg_ddpg'
    env = env_search_control(step_max=200, fuzzy=False, add_noise=False)
    data_path = './prediction_data/'
    model_path = './prediction_model/'
    file_name = '_epochs_5_episodes_100_fuzzy'
    model_name = './prediction_model/'

    train(
        network='mlp',
        env=env,
        data_path=data_path,
        model_based=True,
        memory_extend=True,
        dyna_learning=False,
        model_type='gp',
        noise_type='normal_0.2',
        algorithm_name=algorithm_name,
        file_name=file_name,
        model_path=model_path,
        model_name=model_name,
        restore=False,
        nb_horizon=2,
        nb_epochs=5,
        nb_sample_steps=50,
        nb_samples_extend=10,
        nb_model_learning=30,
        nb_epoch_cycles=100,
        nb_train_steps=60,
        nb_rollout_steps=200
    )




