# -*- coding: utf-8 -*-
"""
# @Time    : 24/10/18 2:40 PM
# @Author  : ZHIMIN HOU
# @FileName: plot_result.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import matplotlib.pyplot as plt
import numpy as np

"""=================================Plot result====================================="""
FORCE_LABELS = ['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$']
POSITION_LABELS = ['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$']
FORCE_Y = 'Force(N)/Moment(10 x Nm)'
POSITION_Y = 'Actions(mm/$\circ$ X10 )'
# ['Fx(N)', 'Fy(N)', 'Fz(N)', 'Mx(Nm)', 'My(Nm)', 'Mz(Nm)']
Title = ["X axis force", "Y axis force", "Z axis force",
         "X axis moment", "Y axis moment", "Z axis moment"]
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']
FONT_SIZE = 34
"""================================================================================="""
PREDICTION_LABELS = ['model-based-ddpg', 'ddpg', 'Prediction-based-ddpg', 'dyna-ddpg']

""" chinese """
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False


def plot(result_path0, result_path1, result_path2, result_path3,
         file_name='./figure/comapre_different_options_reward.pdf'):
    plt.figure(figsize=(10, 8), dpi=300)
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.16, bottom=0.13, right=0.98, top=0.98, wspace=0.23, hspace=0.23)

    # plt.title('Compare Single DDPG with Six Options', fontsize=30)

    prediction_result_0 = np.load(result_path0)
    prediction_result_1 = np.load(result_path1)
    prediction_result_2 = np.load(result_path2)
    prediction_result_3 = np.load(result_path3)

    plt.plot(prediction_result_0, linewidth=4., label='Six-options')
    plt.plot(prediction_result_1, linewidth=4., label='Two-options')
    plt.plot(prediction_result_2, linewidth=4., label='Single-option')
    plt.plot(prediction_result_3, linewidth=4., label='Admittance control')

    # plt.legend(labels=[], loc=2, bbox_to_anchor=(0.1, 1.15),
    #            borderaxespad=0., ncol=3, fontsize=30)

    plt.legend(fontsize=30, labels=PREDICTION_LABELS)
    plt.xlabel("Episodes", fontsize=34)
    plt.ylabel("Episode Reward", fontsize=34)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)

    plt.savefig(file_name)

    plt.show()


def plot_force(forces):
    plt.figure(figsize=(15, 10), dpi=300)
    # plt.title("Contact forces of one episode")
    new_forces = np.zeros((len(forces), 6), dtype=np.float32)
    for i in range(len(forces)):
        new_forces[i, :] = forces[i][0][:6]

    for j in range(6):
        plt.plot(new_forces[:, j], linewidth=3.)

    plt.xlabel("Steps", fontsize=30)
    plt.ylabel("Contact forces: F(N)/M(Nm)", fontsize=30)

    plt.legend(labels=['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'], loc=2, bbox_to_anchor=(0.1, 1.15),
               borderaxespad=0., fontsize=30, ncol=3)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.savefig('./figure/single_ddpg_forces_end.pdf')
    plt.show()


def plot_force_with_variance(forces, file_name, num):

    length = num

    v_forces = np.zeros([len(forces), length, 6])
    for i in range(len(forces)):
        for j in range(length):
            v_forces[i, j, :] = forces[i][j][1][:6]

    mean_force = np.mean(v_forces, axis=0)
    std_force = np.std(v_forces, axis=0)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.tight_layout(pad=4.9, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.17, bottom=0.13, right=0.98, top=0.88, wspace=0.23, hspace=0.23)
    plt.subplot(1, 1, 1)

    for num in range(6):
        if num > 2:
            plt.plot((mean_force[:, num]) * 10, linewidth=3.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             (mean_force[:, num] - std_force[:, num]) * 10,
                             (mean_force[:, num] + std_force[:, num]) * 10, alpha=0.3)
        else:
            plt.plot(mean_force[:, num], linewidth=3.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             (mean_force[:, num] - std_force[:, num]),
                             (mean_force[:, num] + std_force[:, num]), alpha=0.3)

    plt.xlabel("Steps", fontsize=FONT_SIZE)
    plt.ylabel(POSITION_Y, fontsize=FONT_SIZE)
    plt.legend(labels=POSITION_LABELS, loc=2, bbox_to_anchor=(0.1, 1.15),
               borderaxespad=0., ncol=3, fontsize=30)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    plt.savefig(file_name)
    plt.show()


def plot_force_and_moment(path_2, path_3):
    V_force = np.load(path_2)
    V_state = np.load(path_3)
    initial_position = np.array([539.88427, -38.68679, 190.03184, 179.88444, 1.30539, 0.21414])

    v_forces = np.zeros([len(V_force), 15, 6])
    for i in range(len(V_force)):
        for j in range(15):
            for m in range(6):
                v_forces[i, j, m] = np.array(V_force[i])[j, m]

    v_states = np.zeros([len(V_state), 15, 6])
    for i in range(len(V_state)):
        for j in range(15):
            for m in range(6):
                v_states[i, j, m] = np.array(V_state[i])[j, m]

    mean_force = np.mean(v_forces, axis=0)
    std_force = np.std(v_forces, axis=0)
    mean_state = np.mean(v_states, axis=0)
    std_state = np.std(v_states, axis=0)

    plt.figure(figsize=(20, 10), dpi=100)
    plt.subplot(1, 2, 1)
    plt.title("Search Result of Force", fontsize=22)
    for num in range(6):
        plt.plot(mean_force[:, num], linewidth=2.)
        plt.fill_between(np.arange(len(mean_force[:, 0])), mean_force[:, num] - std_force[:, num],
                         mean_force[:, num] + std_force[:, num], alpha=0.3)

    plt.xlabel("Steps", fontsize=22)
    plt.ylabel("F(N)", fontsize=22)
    plt.legend(labels=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], loc='best', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.subplot(1, 2, 2)
    plt.title("Search Result of State", fontsize=22)

    for num in range(6):
        plt.plot(mean_state[:, num] - initial_position[num], linewidth=2.)
        plt.fill_between(np.arange(len(mean_state[:, 0])), mean_state[:, num] - std_state[:, num] - initial_position[num],
                         mean_state[:, num] + std_state[:, num] - initial_position[num], alpha=0.3)
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Coordinate", fontsize=20)
    plt.legend(labels=['x', 'y', 'z', 'rx', 'ry', 'rz'], loc='best', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.savefig('single_force_moment_before_training.pdf')
    plt.show()


def plot_learning_force_and_moment(path_2, path_3, name):

    V_force = np.load(path_2)
    V_state = np.load(path_3)

    initial_position = np.array([539.88427, -38.68679, 190.03184, 179.88444 - 180, 1.30539, 0.21414])

    high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 5])
    low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -5])
    scale = high - low

    # V_force = V_force[80:]
    # V_state = V_state[80:]
    length = 27
    v_forces = np.zeros([len(V_force), length, 6])
    for i in range(len(V_force)):
        for j in range(length):
            for m in range(6):
                v_forces[i, j, m] = np.array(V_force[i])[j, m]

    v_states = np.zeros([len(V_state), length, 6])
    for i in range(len(V_state)):
        for j in range(length):
            for m in range(6):
                v_states[i, j, m] = np.array(V_state[i])[j, m+6]
    mean_force = np.mean(v_forces, axis=0)
    std_force = np.std(v_forces, axis=0)
    mean_state = np.mean(v_states, axis=0)
    std_state = np.std(v_states, axis=0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    plt.figure(figsize=(20, 10), dpi=100)

    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.98, wspace=0.23, hspace=0.23)
    plt.subplot(1, 2, 1)
    for num in range(6):
        if num > 2:
            plt.plot((mean_force[:, num] * scale[num] + low[num]) * 10, linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             ((mean_force[:, num] - std_force[:, num]) * scale[num] + low[num]) * 10,
                             ((mean_force[:, num] + std_force[:, num]) * scale[num] + low[num]) * 10, alpha=0.3)
        else:
            plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
                             (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)

        # plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.)
        # plt.fill_between(np.arange(len(mean_force[:, 0])),
        #                  (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
        #                  (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)

    plt.xlabel("Steps", fontsize=30)
    plt.ylabel("Forces$(N)$ / Moments$(10XNm)$", fontsize=30)
    plt.legend(labels=['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'], loc='lower right', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.subplot(1, 2, 2)
    for num in range(6):
        plt.plot(mean_state[:, num] * scale[num + 6] + low[num + 6] - initial_position[num], linewidth=2.75)
        plt.fill_between(np.arange(len(mean_state[:, 0])),
                         (mean_state[:, num] - std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num],
                         (mean_state[:, num] + std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num], alpha=0.3)
    plt.xlabel("Steps", fontsize=30)
    plt.ylabel("Position$(mm)$ / Orientation$(\circ)$", fontsize=30)
    plt.legend(labels=['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$'], loc='lower right', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.savefig(name + '.pdf')
    plt.show()


def plot_chinese_learning_force_and_moment(path_2, path_3, name):

    font_size = 30
    V_force = np.load(path_2)
    V_state = np.load(path_3)

    initial_position = np.array([539.88427, -38.68679, 190.03184, 179.88444 - 180, 1.30539, 0.21414])

    high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 5])
    low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -5])
    scale = high - low
    length = 27
    v_forces = np.zeros([len(V_force), length, 6])
    for i in range(len(V_force)):
        for j in range(length):
            for m in range(6):
                v_forces[i, j, m] = np.array(V_force[i])[j, m]

    v_states = np.zeros([len(V_state), length, 6])
    for i in range(len(V_state)):
        for j in range(length):
            for m in range(6):
                v_states[i, j, m] = np.array(V_state[i])[j, m+6]

    mean_force = np.mean(v_forces, axis=0)
    std_force = np.std(v_forces, axis=0)

    mean_state = np.mean(v_states, axis=0)
    std_state = np.std(v_states, axis=0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    plt.figure(figsize=(10, 8), dpi=300)

    # plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.tight_layout(pad=4.9, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.14, bottom=0.12, right=0.98, top=0.88, wspace=0.23, hspace=0.23)
    plt.subplot(1, 1, 1)
    for num in range(6):
        if num > 2:
            plt.plot((mean_force[:, num] * scale[num] + low[num]) * 10, linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             ((mean_force[:, num] - std_force[:, num]) * scale[num] + low[num]) * 10,
                             ((mean_force[:, num] + std_force[:, num]) * scale[num] + low[num]) * 10, alpha=0.3)
        else:
            plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
                             (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)

        # plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.)
        # plt.fill_between(np.arange(len(mean_force[:, 0])),
        #                  (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
        #                  (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)

    # plt.xlabel("Steps", fontsize=30)
    # plt.ylabel("Forces$(N)$ / Moments$(10XNm)$", fontsize=30)
    # plt.legend(labels=['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'], loc='lower right', fontsize=30)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)

    # chinese
    plt.xlabel("装配步数", fontsize=font_size)
    plt.ylabel("接触力(N)/接触力矩(10XNm)", fontsize=font_size)
    plt.legend(labels=['$F^x$', '$F^y$', '$F^z$', '$M^x$', '$M^y$', '$M^z$'], loc=2, bbox_to_anchor=(0.1, 1.15), borderaxespad=0., fontsize=30, ncol=3)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # plt.subplot(1, 2, 2)

    # for num in range(6):
    #     plt.plot(mean_state[:, num] * scale[num + 6] + low[num + 6] - initial_position[num], linewidth=2.75)
    #     plt.fill_between(np.arange(len(mean_state[:, 0])),
    #                      (mean_state[:, num] - std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num],
    #                      (mean_state[:, num] + std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num], alpha=0.3)

    # # plt.xlabel("Steps", fontsize=30)
    # # plt.ylabel("Position$(mm)$ / Orientation$(\circ)$", fontsize=30)
    # # plt.legend(labels=['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$'], loc='lower right', fontsize=30)
    # # plt.xticks(fontsize=30)
    # # plt.yticks(fontsize=30)
    #
    # chinese
    # plt.xlabel("装配步数", fontsize=font_size)
    # plt.ylabel("双轴位置(mm)/姿态($\circ$)", fontsize=font_size)
    # plt.legend(labels=['$P^x$', '$P^y$', '$P^z$', '$O^x$', '$O^y$', '$O^z$'], loc=2, bbox_to_anchor=(0.1, 1.15), borderaxespad=0., fontsize=30, ncol=3)
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)

    plt.savefig('./figure/pdf/single_chinese_' + name + '_force_moment.pdf')
    # plt.savefig('./figure/pdf/single_chinese_' + name + '_position_orientation.pdf')
    # plt.show()


def plot_compare(fuzzy_path, none_fuzz_path):

    reward_fuzzy = np.load(fuzzy_path)
    reward_none_fuzzy = np.load(none_fuzz_path)
    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.9, w_pad=1., h_pad=1.)

    CHINESE = False

    if CHINESE:
        plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_fuzzy[1:]), color='b', linewidth=2.5,
                 label='DDPG')
        plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_none_fuzzy[1:]), color='r', linewidth=2.5,
                 label='基于环境预测知识优化DDPG')
        plt.ylabel('每回合装配累计奖励值', fontsize=28)
        plt.xlabel('训练回合数', fontsize=28)
    else:

        plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_fuzzy[1:]), color='b',
                 linewidth=2.5, label='Normal DDPG')
        plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_none_fuzzy[1:]), color='r',
                 linewidth=2.5, label='Prediction-based DDPG')
        plt.ylabel('Episode Step', fontsize=28)
        plt.xlabel('Episodes', fontsize=28)


    # plt.ylim(-5, 1)
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    # plt.ylabel('每回合装配步数', fontsize=28)
    plt.legend(fontsize=28, loc='best')

    # plot_reward('./episode_rewards_100.npy')
    # plt.figure(figsize=(15, 15), dpi=100)
    # plt.subplot(2, 1, 2)
    # plt.title('DQN With Knowledge')
    # plt.plot(np.arange(len(reward_fuzzy) - 1), np.array(reward_fuzzy[1:] * 10), color='b')
    # plt.ylabel('Episode Reward', fontsize=20)
    # plt.xlabel('Episodes', fontsize=20)

    # plt.savefig('./figure/pdf/chinese_ddpg_episode_reward.pdf')
    # plt.savefig('./figure/jpg/chinese_ddpg_episode_reward.jpg')

    # plt.savefig('./figure/pdf/chinese_ddpg_episode_step.pdf')
    # plt.savefig('./figure/jpg/chinese_ddpg_episode_step.jpg')

    plt.show()


def new_plot_compare(fuzzy_path, none_fuzz_path):

    reward_fuzzy = np.load(fuzzy_path)
    reward_none_fuzzy = np.load(none_fuzz_path)

    reward_average = np.zeros(len(reward_fuzzy))
    reward_none_average = np.zeros(len(reward_none_fuzzy))
    for i in range(len(reward_fuzzy)):
        sum_1 = 0
        sum_2 = 0
        num = 0
        for j in range(10):
            if j + i < len(reward_fuzzy):
                sum_1 += reward_fuzzy[i+j]
                sum_2 += reward_none_fuzzy[i+j]
                num += 1
        reward_average[i] = sum_1/num
        reward_none_average[i] = sum_2/num

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.9, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.88, wspace=0.23, hspace=0.23)
    # plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_average[1:]), color='r', linewidth=5, alpha=0.4)
    # plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_none_average[1:]), color='b', linewidth=5, alpha=0.4)
    # plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_none_fuzzy[1:]), color='b', linewidth=2.5, label='Normal DQN')
    # plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_fuzzy[1:]), color='r', linewidth=2.5, label='Prediction-based DQN')

    plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_none_fuzzy[1:]), color='b', linewidth=2.5,
             label='DDPG')
    plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_fuzzy[1:]), color='r', linewidth=2.5,
             label='基于环境预测知识优化DDPG')

    # plt.ylim(-5, 1)

    # plt.yticks(fontsize=24)
    # plt.xticks(fontsize=24)
    # plt.ylabel('Episode Step', fontsize=24)
    # plt.xlabel('Episodes', fontsize=24)
    # plt.legend(fontsize=24, loc='lower right')

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    plt.xlabel('训练回合数', fontsize=30)
    plt.ylabel('每回合装配累计奖励值', fontsize=30)
    # plt.ylabel('每回合装配步数', fontsize=30)
    plt.legend(fontsize=28, loc=2, bbox_to_anchor=(0.15, 1.15), borderaxespad=0)

    # plot_reward('./episode_rewards_100.npy')
    # plt.figure(figsize=(15, 15), dpi=100)
    # plt.subplot(2, 1, 2)
    # plt.title('DQN With Knowledge')
    # plt.plot(np.arange(len(reward_fuzzy) - 1), np.array(reward_fuzzy[1:] * 10), color='b')
    # plt.ylabel('Episode Reward', fontsize=20)
    # plt.xlabel('Episodes', fontsize=20)

    # plt.savefig('./figure/pdf/chinese_ddpg_episode_step.pdf')
    # plt.savefig('./figure/jpg/chinese_ddpg_episode_step.jpg')

    # plt.savefig('./figure/pdf/chinese_ddpg_episode_reward.pdf')
    # plt.savefig('./figure/jpg/chinese_ddpg_episode_reward.jpg')

    plt.show()


def plot_continuous_data(path):
    raw_data = np.load(path)
    print(raw_data)
    plt.figure(figsize=(20, 15))
    plt.title('Episode Reward')
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.9, wspace=0.23, hspace=0.22)
    # plt.subplots_adjust(left=0.065, bottom=0.1, right=0.995, top=0.9, wspace=0.2, hspace=0.2)
    data = np.zeros((len(raw_data), 12))
    for j in range(len(raw_data)):
        data[j] = raw_data[j, 0]
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.plot(data[:, j], linewidth=2.5, color='r')
        # plt.ylabel(YLABEL[j], fontsize=18)
        if j>2:
            plt.xlabel('steps', fontsize=30)
        plt.title(YLABEL[j],fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
    # plt.savefig('raw_data.pdf')
    plt.show()


def plot_comparision_hist(result_path_1, result_path_2, file_name):

    result_data_1 = np.load(result_path_1)
    result_data_2 = np.load(result_path_2)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.hist(result_data_1, bins=40, histtype="stepfilled", label='After Training')
    plt.hist(result_data_2, bins=40, histtype="stepfilled", label='Before Training', color='red')

    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(np.arange(10., 35, 5), fontsize=FONT_SIZE)
    plt.ylabel('Frequency', fontsize=FONT_SIZE)
    plt.xlabel('Episode time(s)', fontsize=FONT_SIZE)
    plt.grid(axis="y")
    plt.legend(fontsize=30, loc='best')

    plt.savefig(file_name)
    plt.show()


if __name__ == "__main__":
    # samples = np.load('./second_data/train_step_none_fuzzy.npy')
    # print(samples)
    # plot_compare('./episode_rewards_fuzzy.npy', './episode_rewards_none_fuzzy.npy')
    # plot('./second_data/train_step_fuzzy_1.npy')
    # plot('./fourth_data/train_reward_none_fuzzy_none_model_test_normal_0.2_episodes_150.npy')
    # print(data)
    # plot('./fourth_data/cc',
    #      './fourth_data/train_reward_none_fuzzy_none_model_ddpg_normal_0.2_episodes_100.npy')
    # plot('./fifth_data/train_reward_option_normal_0.2_episodes_150.npy',
    #      './fifth_data/train_reward_2_option_normal_0.2_episodes_150.npy',
    #      './fifth_data/train_reward_1_option_normal_0.2_episodes_150.npy',
    #      './fifth_data/train_reward_pdc_normal_0.2_episodes_150.npy')

    plot('train_reward_mb_ddpg_normal_0.2_episodes_100_none_fuzzy.npy',
         'train_reward_ddpg_normal_0.2_episodes_100_none_fuzzy.npy',
         'train_reward_ddpg_normal_0.2_episodes_100_fuzzy.npy',
         'train_reward_dyna_ddpg_normal_0.2_episodes_100_none_fuzzy.npy',
         file_name='./figures/mb_ddpg.pdf')

    # data = np.load('./pdc_data/train_states_pdc_normal_0.2_episodes_150.npy')
    # print(data[0])

    # forces = np.load('./pdc_data/train_states_pdc_normal_0.2_episodes_150.npy')
    # forces = np.load('./two_options_data/train_states_2_option_test_normal_0.2_episodes_150.npy')
    # forces = np.load('./six_options_data/train_states_6_option_test_normal_0.2_episodes_150.npy')
    # print(forces[0][0][0][:6])

    # plot_force_with_variance(forces, 'actions_before_training.pdf', 127)

    # print(len(forces))
    #
    # num = np.zeros(len(forces))
    # print(len(forces))
    # for i in range(len(forces)):
    #     num[i] = len(forces[i])
    # #
    # print(min(num))

    # data_1 = np.load('./six_options_data/train_times_6_options_new_test_normal_0.2_episodes_20.npy')
    # print(data_1)
    # plot_comparision_hist('./six_options_data/train_times_6_options_new_test_normal_0.2_episodes_20.npy',
    #                           './pdc_data/train_times_pdc_normal_0.2_episodes_150.npy', 'comparision_episode_time.pdf')

    #
    # # print(forces[149][0][0][:6])
    # plot_force(len(forces[0]))

    # high = np.array([50, 50, 0, 5, 5, 6, 542, -36, 192, 5, 5, 6])
    # low = np.array([-50, -50, -50, -5, -5, -6, 538, -42, 188, -5, -5, -6])
    # scale = np.array([100, 100, 50, 10, 10, 12, 4, 6, 4, 10, 10, 12])
    #
    # episodes = np.zeros((len(data[0]), 12), dtype=np.float32)
    # for j in range(len(data[0])):
    #     episodes[j, :] = data[0][j][0]
    # print('episodes', episodes[0:2])
    # print(np.max(episodes[0:2], axis=0))
    #
    # for i in range(1, len(data)):
    #     episode = np.zeros((len(data[i]), 12), dtype=np.float32)
    #     for j in range(len(data[i])):
    #         episode[j, :] = data[i][j][0] * scale + low
    #     episodes = np.append(episodes, episode, axis=0)

    # print(np.max(episodes, axis=1))

    # plot_force_and_moment('./search_force_noise.npy', './search_state_noise.npy')
    # plot_learning_force_and_moment('./train_states_none_fuzzy.npy', './train_states_none_fuzzy.npy', 'ddpg_none_fuzzy')
    # plot_learning_force_and_moment('./test_states_fuzzy.npy', './test_states_fuzzy.npy', 'ddpg_fuzzy')

    # plot_chinese_learning_force_and_moment('./train_states_none_fuzzy.npy', './train_states_none_fuzzy.npy', 'ddpg_none_fuzzy')
    # plot_chinese_learning_force_and_moment('./test_states_fuzzy.npy', './test_states_fuzzy.npy', 'ddpg_fuzzy')

    # reward = np.load('reward.npy')
    # plot_reward('step.npy')
    # plot_reward('none_reward.npy')
    # plot_compare('reward.npy', 'none_reward.npy')

    # plot_force_and_moment('none_states.npy')
    # plot_compare('reward.npy', 'none_reward.npy')
    # plot_compare('step.npy', 'none_step.npy')

    # new_plot_compare('none_step.npy', 'step.npy')
    # new_plot_compare('none_reward.npy', 'reward.npy')

    # plot_comparision_hist('step.npy', 'none_step.npy')
    # plot_comparision_hist('reward.npy', 'none_reward.npy')
    # plot_comparision_hist('./test_states_fuzzy.npy', './train_states_none_fuzzy.npy')
    # print(len(np.load('./train_states_none_fuzzy.npy')[0]))
