from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import matplotlib.pyplot as plt

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd

LAYER1 = 32

EVAL_SCOPE = 'eval_upper_critic'
TARGET_SCOPE = 'target_upper_critic'

INIT_WEIGHT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
INIT_BIAS = tf.constant_initializer(0.1)


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(
                tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class UCritic:

    def __init__(self, session, state_dim, option_num, gamma, epsilon, tau=1e-2, learning_rate=1e-3):
        """Initiate the critic network for normalized states and options"""

        # tensorflow session
        self.sess = session

        # environment parameters
        self.sd = state_dim
        self.on = option_num
        self.eps = epsilon

        # some placeholder
        self.r = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='upper_reward')
        self.o = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='option')

        # evaluation and target network
        self.s, self.q = self._q_net(scope=EVAL_SCOPE, trainable=True)
        self.s_, q_ = self._q_net(scope=TARGET_SCOPE, trainable=False)

        # index the q and get max q_
        qo = tf.gather_nd(params=self.q,
                          indices=tf.stack([tf.range(tf.shape(self.o)[0], dtype=tf.int32),
                                            self.o[:, 0]], axis=1))
        qm = self.r + gamma * tf.reduce_max(q_, axis=1)

        # soft update
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=EVAL_SCOPE)
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=TARGET_SCOPE)
        self.update = [tf.assign(t, t + tau * (e - t)) for t, e in zip(target_params, eval_params)]

        # define the error and optimizer
        self.loss = tf.losses.mean_squared_error(labels=qm[:, 0], predictions=qo)
        self.op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=eval_params)

        # pretrain placeholder
        self.tru = tf.placeholder(dtype=tf.int32, shape=(None, self.on))
        self.lo = tf.losses.hinge_loss(labels=self.tru, logits=self.q)
        self.pop = tf.train.AdamOptimizer(1e-3).minimize(self.lo)

    def train(self, state_batch, option_batch, reward_batch, next_state_batch):
        """Train the critic network"""

        # minimize the loss
        self.sess.run(self.op, feed_dict={
            self.s: state_batch,
            self.o: option_batch,
            self.r: reward_batch,
            self.s_: next_state_batch
        })

        # target update
        self.sess.run(self.update)

    def choose_option(self, state):
        """Get the q batch"""

        if np.random.rand() < self.eps:
            return np.random.choice(self.on)
        else:
            return self.sess.run(self.q, feed_dict={
                self.s: state[np.newaxis, :]
            })[0].argmax()

    def get_distribution(self, state_batch):
        """Get the option distribution"""

        # get q batch
        res = self.sess.run(self.q, feed_dict={
            self.s: state_batch
        })
        index = np.argmax(res, axis=1)
        res = np.ones(res.shape) * self.eps / self.on
        res[np.arange(res.shape[0], dtype=np.int32), index] += 1 - self.eps

        return res

    def pretrain(self):
        """Pretrain upper critic"""

        num = 50
        delta = 2.0 / (num - 1)
        test_state = -np.ones((num * num, 2))
        test_label = np.zeros((num * num, 4), dtype=np.int32)
        labels = np.eye(4, dtype=np.int32)
        for i in range(num):
            for j in range(num):
                o = i * num + j
                s = np.array([i * delta, j * delta])
                test_state[o] += s
                if test_state[o, 0] > 0 and test_state[o, 1] > 0:
                    test_label[o] = labels[0]
                elif test_state[o, 0] < 0 < test_state[o, 1]:
                    test_label[o] = labels[1]
                elif test_state[o, 0] < 0 and test_state[o, 1] < 0:
                    test_label[o] = labels[2]
                elif test_state[o, 1] < 0 < test_state[o, 0]:
                    test_label[o] = labels[3]

        while True:
            self.sess.run(self.pop, feed_dict={
                self.s: test_state,
                self.tru: test_label
            })
            a = self.sess.run(self.lo, feed_dict={
                self.s: test_state,
                self.tru: test_label
            })
            if a < 1e-3:
                break

    def _q_net(self, scope, trainable):
        """Generate evaluation/target q network"""

        with tf.variable_scope(scope):

            state = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='state')

            x = tf.layers.dense(state, LAYER1, activation=tf.nn.relu,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='dense1')

            q = tf.layers.dense(x, self.on, activation=None,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='layer2')

        return state, q

    def render(self):
        """Render the critic network"""

        if self.sd == 2:
            num = 50
            X = np.linspace(-1.0, 1.0, num)
            Y = np.linspace(-1.0, 1.0, num)
            C = np.zeros((num, num))

            s = np.zeros((num, 2))
            for i in range(num):
                s[:, 0] = X[i]
                s[:, 1] = Y
                C[:, i] = self.sess.run(self.q, feed_dict={
                    self.s: s
                }).argmax(axis=1)
            im = plt.pcolor(X, Y, C, cmap='jet')
            plt.title('critic output')
            plt.colorbar(im)
            plt.show()


class DDPG(object):
    def __init__(self, actors, critic, memorys, observation_shape, action_shape, param_noise=None, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
                 adaptive_param_noise=True, adaptive_param_noise_policy_threshold=.1,
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):

        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + (action_shape,), name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memorys = memorys
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actors = actors

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actors = [copy(actor) for actor in actors]
        for target_actor in target_actors:
            target_actor.name = 'target_' + target_actor.name
        self.target_actors = target_actors

        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.actors_tf = [actor(normalized_obs0) for actor in actors]
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)

        self.critic_tf = denormalize(
            tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        self.normalized_critic_with_actors_tf = [critic(normalized_obs0, actor_tf, reuse=True) for actor_tf in self.actors_tf]
        self.critic_with_actors_tf = [denormalize(
            tf.clip_by_value(normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
            self.ret_rms) for normalized_critic_with_actor_tf in self.normalized_critic_with_actors_tf]

        Q_obs1s = [denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms) for target_actor in target_actors]
        self.target_Qs = [self.rewards + (1. - self.terminals1) * gamma * Q_obs1 for Q_obs1 in Q_obs1s]

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        self.setup_actors_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        # self.setup_stats()
        self.setup_target_network_updates()

        # recurrent architectures not supported yet
        self.initial_state = None
        self.saver = tf.train.Saver()

    def setup_target_network_updates(self):

        actors_init_updates = []
        actors_soft_updates = []
        for i in range(len(self.actors)):
            actor_init_updates, actor_soft_updates = get_target_updates(self.actors[i].vars, self.target_actors[i].vars, self.tau)
            actors_init_updates.append(actor_init_updates)
            actors_soft_updates.append(actor_soft_updates)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars,
                                                                      self.tau)
        self.target_init_updates = actors_init_updates + [critic_init_updates]
        self.target_soft_updates = [[actors_soft_updates[i], critic_soft_updates] for i in range(len(self.actors))]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actors = [copy(actor) for actor in self.actors]
        for param_noise_actor in param_noise_actors:
            param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actors_tf = [param_noise_actor(normalized_obs0) for param_noise_actor in param_noise_actors]
        logger.info('setting up param noise')
        self.perturb_policy_ops = [get_perturbed_actor_updates(self.actors[i], param_noise_actors[i], self.param_noise_stddev) for i in range(len(self.actors))]

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actors = [copy(actor) for actor in self.actors]
        for adaptive_param_noise_actor in adaptive_param_noise_actors:
            adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actors_tf = [adaptive_param_noise_actor(normalized_obs0) for adaptive_param_noise_actor in adaptive_param_noise_actors]
        self.perturb_adaptive_policy_ops = [get_perturbed_actor_updates(self.actors[i], adaptive_param_noise_actors[i],
                                                                       self.param_noise_stddev) for i in range(len(self.actors))]
        #self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actors_optimizer(self):
        logger.info('setting up actor optimizer')

        self.actors_loss = []
        self.actors_grads = []
        self.actors_optimizer = []

        for i in range(len(self.actors)):
            actor_loss = -tf.reduce_mean(self.critic_with_actors_tf[i])
            actor_shapes = [var.get_shape().as_list() for var in self.actors[i].trainable_vars]
            actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])

            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
            actor_grads = U.flatgrad(actor_loss, self.actors[i].trainable_vars, clip_norm=self.clip_norm)
            actor_optimizer = MpiAdam(var_list=self.actors[i].trainable_vars,
                                           beta1=0.9, beta2=0.999, epsilon=1e-08)

            self.actors_loss.append(actor_loss)
            self.actors_grads.append(actor_grads)
            self.actors_optimizer.append(actor_optimizer)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if
                               'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))

        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
                                        beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.

        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean
        self.renormalize_Q_outputs_op = []

        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def step(self, obs, sigma, index, apply_noise=True, compute_Q=True):

        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actors_tf[index]
        else:
            actor_tf = self.actors_tf[index]
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [obs])}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actors_tf[index]], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise(sigma)
            """assert noise.shape == action.shape"""
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q, None, None

    def store_transition(self, obs0, action, reward, obs1, terminal1, index):
        reward *= self.reward_scale
        # B = obs0.shape[0]
        # for b in range(B):
        #     self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])
        #     if self.normalize_observations:
        #         self.obs_rms.update(np.array([obs0[b]]))
        self.memorys[index].append(obs0, action, reward, obs1, terminal1)

    def train(self, index):
        # Get a batch.
        batch = self.memorys[index].sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q],
                                                        feed_dict={
                                                            self.obs1: batch['obs1'],
                                                            self.rewards: batch['rewards'],
                                                            self.terminals1: batch['terminals1'].astype('float32'),
                                                        })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })

        else:
            target_Q = self.sess.run(self.target_Qs[index], feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.actors_grads[index], self.actors_loss[index], self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })

        self.actors_optimizer[index].update(actor_grads, stepsize=self.actor_lr)

        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        for ao in self.actors_optimizer:
            ao.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self, index):
        self.sess.run(self.target_soft_updates[index])

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.actions: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = distance
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        """Reset internal state after an episode is complete."""

        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def store(self, path):

        self.saver = self.saver.save(self.sess, path)

    def restore(self, sess, path, name):

        self.saver = tf.train.import_meta_graph(path + name + '.meta')
        # self.saver.restore(sess, tf.train.latest_checkpoint(path))
        self.saver.restore(sess, path + name)
        self.sess = sess
