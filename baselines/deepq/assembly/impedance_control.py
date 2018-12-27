# -*- coding: utf-8 -*-
"""
# @Time    : 27/12/18 12:45 PM
# @Author  : ZHIMIN HOU
# @FileName: impedance_control.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

from baselines.deepq.assembly.Env_robot_control import env_single_insert_control
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=['device', 'file'])
    parser.add_argument('--path', type=str)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=3000)
    parser.add_argument('--data-file', type=str)
    parser.add_argument('--lambda', type=float, default=0.6)
    parser.add_argument('--meta_step_size', type=float, default=0.00001)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--loop', type=float, default=1)
    parser.add_argument('--noplot', action='store_false', dest='plot')
    parser.add_argument('--record-file', type=str)
    parser.add_argument('--seed', type=int)
    return vars(parser.parse_args())


args = parse_args()
env = env_single_insert_control()
env.pull_peg_up()
obv = env.reset()
print(obv)

for i in range(args['episodes']):

    done = env.force_control(env.desired_force_moment[0], obv[:6], obv[6:], i)

    obv = env.get_state()

    if done:
        env.pull_peg_up()