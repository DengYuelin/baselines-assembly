import os
import time
from collections import deque
import pickle

import gym
from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.deepq.assembly.Env_robot_control import env_search_control, env_continuous_search_control

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression

import baselines.common.tf_util as U

from baselines import logger
import numpy as np
from mpi4py import MPI
import copy as cp


def learn(network,
          env,
          path='',
          model_path='./model/',
          model_name='ddpg_none_fuzzy',
          file_name='test',
          model_based=False,
          model_type='gp',
          restore=False,
          seed=None,
          total_timesteps=None,
          nb_epochs=1,  # with default settings, perform 1M steps total
          nb_epoch_cycles=1,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2', #'adaptive-param_0.2', ou_0.2, normal_0.2
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50,  # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=32,  # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          **network_kwargs):

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 1

    rank = MPI.COMM_WORLD.Get_rank()
    nb_actions = env.action_space.n
    # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

    memory = Memory(limit=int(1e5), action_shape=env.action_space.n, observation_shape=env.observation_space.shape)
    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    action_noise = None
    param_noise = None
    nb_actions = env.action_space.n

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

    if model_based:
        if model_type == 'gp':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            dynamic_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        elif model_type == 'linear':
            dynamic_model = LinearRegression()
        else:
            dynamic_model = LinearRegression()

    max_action = env.max_action
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.n,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm, reward_scale=reward_scale)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    sess = U.get_session()

    if restore:
        agent.restore(sess, model_path, model_name)
    else:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

    agent.reset()
    episodes = 0
    t = 0
    epoch = 0

    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_episode_states = []
    epoch_qs = []
    epoch_episodes = 0
    num_samples = 5
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            obs, done, state = env.reset()
            episode_reward = 0.
            episode_step = 0
            episode_states = []
            logger.info("================== The {} episode start !!! ===================".format(cycle))
            for t_rollout in range(nb_rollout_steps):

                # Predict next action.
                action, q, _, _ = agent.step(obs, stddev, apply_noise=True, compute_Q=True)

                # Execute next action.
                if rank == 0 and render:
                    env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                new_obs, r, done, safe_or_not, final_action, new_state = env.step(max_action * action, t_rollout)

                if safe_or_not is False:
                    break

                t += 1
                if rank == 0 and render:
                    env.render()

                episode_reward += r
                episode_step += 1
                episode_states.append([cp.deepcopy(obs), cp.deepcopy(final_action), np.array(cp.deepcopy(r)), cp.deepcopy(new_obs)])

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)

                agent.store_transition(obs, action, r, new_obs, done)

                if model_based and cycle > 3:
                    pred_x = np.zeros((1, 18), dtype=np.float32)
                    for j in range(num_samples):
                        m_action, _, _, _ = agent.step(obs, stddev, apply_noise=True, compute_Q=False)
                        pred_x[:, :12] = obs
                        pred_x[:, 12:] = m_action
                        m_new_obs = dynamic_model.predict(pred_x)[0]
                        state = env.inverse_state(m_new_obs)
                        m_reward = env.get_reward(state, m_action)
                        agent.store_transition(obs, m_action, m_reward, m_new_obs, done)

                # the batched data will be unrolled in memory.py's append.
                obs = new_obs

                # for d in range(len(done)):
                #     if done[d]:
                #         # Episode done.
                #         epoch_episode_rewards.append(episode_reward[d])
                #         episode_rewards_history.append(episode_reward[d])
                #         epoch_episode_steps.append(episode_step[d])
                #         episode_reward[d] = 0.
                #         episode_step[d] = 0
                #         epoch_episodes += 1
                #         episodes += 1
                #         if nenvs == 1:
                #             agent.reset()

                if done:
                    # Episode done.
                    # episode_reward = 0.
                    # episode_step = 0.

                    # if nenvs == 1:
                    #     agent.reset()
                    break

            if model_based and cycle > 2:
                # input_x = np.zeros((len(episode_states), 18), dtype=np.float32)
                # input_y = np.zeros((len(episode_states), 12), dtype=np.float32)
                # for i in range(len(episode_states)):
                #     input_x[i, :12] = episode_states[i][0]
                #     input_x[i, 12:] = episode_states[i][1]
                #     input_y[i, :] = episode_states[i][3]
                # if cycle == 0 and epoch == 0:
                #     train_x = input_x
                #     train_y = input_y
                # else:
                #     train_x = np.append(train_x, input_x, axis=0)
                #     train_y = np.append(train_y, input_y, axis=0)
                # dynamic_model.fit(train_x, train_y)
                for i in range(len(epoch_episode_states)):
                    input_x = np.zeros((len(epoch_episode_states[i]), 18), dtype=np.float32)
                    input_y = np.zeros((len(epoch_episode_states[i]), 12), dtype=np.float32)
                    for j in range(len(epoch_episode_states[i])):
                        input_x[j, :12] = epoch_episode_states[i][j][0]
                        input_x[j, 12:] = epoch_episode_states[i][j][1]
                        input_y[j, :] = epoch_episode_states[i][j][3]

                dynamic_model.fit(input_x, input_y)

            stddev = float(stddev) * 0.95
            epoch_episode_rewards.append(episode_reward)
            episode_rewards_history.append(episode_reward)
            epoch_episode_steps.append(episode_step)
            epoch_episode_states.append(cp.deepcopy(episode_states))
            epoch_episodes += 1
            episodes += 1
            logger.info("================== The {} steps finish  !!! ===================".format(t_rollout))

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):

                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)
                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype=np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0

        mpi_size = MPI.COMM_WORLD.Get_size()

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)

        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)

        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([ np.array(x).flatten()[0] for x in combined_stats.values()]))
        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()

        # # save data
        np.save(path+'train_reward_none_fuzzy_none_model_test_' + noise_type, epoch_episode_rewards)
        np.save(path+'train_step_none_fuzzy_none_model_test_' + noise_type, epoch_episode_steps)
        np.save(path+'train_states_none_fuzzy_none_model_test_' + noise_type, epoch_episode_states)

        # # agent save
        agent.store(model_path + 'train_model_none_fuzzy_none_model_test_' + noise_type)


if __name__ == '__main__':

    # env = env_search_control()
    env = env_continuous_search_control(fuzzy=False)

    # env = gym.make("HalfCheetah-v2")
    path = './data/third_data/'
    model_path = './data/third_model/'
    learn(network='mlp',
          env=env,
          nb_epoch_cycles=1,
          path=path,
          model_based=False,
          model_type='linear',
          noise_type='normal_0.2',
          file_name='model',
          model_path=model_path,
          nb_train_steps=50,
          nb_rollout_steps=400)