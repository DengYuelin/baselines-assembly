#!/usr/bin/env python -O
# -*- coding: ascii -*-

import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt

#(state, gamma, np.array(action), next_state)
#data_second = np.load('./data/data_prediction_random_initial_position_first.npy')
#data_third = np.load('./data/data_prediction_random_initial_position_second.npy')
#V_force = np.load('./data/search_force_1.npy')
#V_state = np.load('./data/search_state_1.npy')

#a = np.vstack((data_second, data_third))

"""================================================================================="""

"""=================================Plot result====================================="""
YLABEL = ['Fx(N)', 'Fy(N)', 'Fz(N)', 'Mx(Nm)', 'My(Nm)', 'Mz(Nm)']
Title = ["X axis force", "Y axis force", "Z axis force",
         "X axis moment", "Y axis moment", "Z axis moment"]
"""================================================================================="""

"""================================Linear function=================================="""
"""Tile coding"""

def plot_raw_data(path_1):
    data = np.load(path_1)
    force_m = np.zeros((len(data), 12))
  #  V_force_m = np.zeros((len(V_force), 12))
  #  V_state_m = np.zeros((len(V_force), 12))

    plt.figure(figsize=(20, 20), dpi=100)
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.065, bottom=0.1, right=0.995, top=0.9, wspace=0.2, hspace=0.2)
    plt.title("True Data")
    for j in range(len(data)):
        force_m[j] = data[j, 0]
    k = -1
    for i in range(len(data)):
        if data[i, 1] == 0:
            print("===========================================")
            line = force_m[k+1:i+1]
            print(line)
            k = i
            for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.plot(line[:, j])
                # plt.plot(line[:, 0])

                if j == 1:
                    plt.ylabel(YLABEL[j], fontsize=17.5)
                    plt.xlabel('steps', fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)

                else:
                    plt.ylabel(YLABEL[j], fontsize=20)
                    plt.xlabel('steps', fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
        i += 1
   # plt.savefig('raw_data_random_policy1.jpg')


def plot_force_and_moment(path_2, path_3):
    V_force = np.load(path_2)
    V_state = np.load(path_3)
    plt.figure(figsize=(15, 10), dpi=100)
    plt.title("Search Result of Force", fontsize=20)
    plt.plot(V_force[:100])
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("F(N)", fontsize=20)
    plt.legend(labels=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], loc='best', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.figure(figsize=(15, 10), dpi=100)
    plt.title("Search Result of State", fontsize=20)
    plt.plot(V_state[:100] - [539.88427, -38.68679, 190.03184, 179.88444, 1.30539, 0.21414])
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Coordinate", fontsize=20)
    plt.legend(labels=['x', 'y', 'z', 'rx', 'ry', 'rz'], loc='best', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()

if __name__ == '__main__':
    # plot_raw_data('./data/data_prediction_random_initial_position_first.npy', './data/search_force_1.npy', './data/search_state_1.npy')
    # get command line arguments
    #args = parse_args()
    #State, Gamma, Force = get_data(args)
    #print(State.shape)

    #V = true_value(State, 0.99, Gamma, Force[:, 1])
    # print(V)

    # train(args, State, Gamma, Force)

    # linear_function(args, State, Gamma, Force)

    plot_force_and_moment('./search_force.npy', './search_state.npy')
    # print(V)
