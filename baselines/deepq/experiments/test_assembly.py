# -*- coding: utf-8 -*-
"""
# @Time    : 25/10/18 2:32 PM
# @Author  : ZHIMIN HOU
# @FileName: test_assembly.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import numpy as np
from baselines import deepq
from baselines.common import models
from baselines import logger
import copy as cp
from baselines.deepq.assembly.Env_robot_control import env_search_control


def main(
        test_episodes=10,
        test_steps=50
        ):
    env = env_search_control()
    print(env.observation_space)
    print(env.action_space)
    act = deepq.learn(
        env,
        network=models.mlp(num_layers=1, num_hidden=64),
        total_timesteps=0,
        total_episodes=0,
        total_steps=0,
        load_path='assembly_model.pkl'
    )
    episode_rewards = []
    for i in range(test_episodes):
        obs, done = env.reset()
        episode_rew = 0
        logger.info("================== The {} episode start !!! ===================".format(i))
        for j in range(test_steps):
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        episode_rewards.append(cp.deepcopy(episode_rew))
        print("Episode reward", episode_rew)

    np.save('../data/test_episode_reward', episode_rewards)


if __name__ == '__main__':
    main(
        test_episodes=10,
        test_steps=50
    )