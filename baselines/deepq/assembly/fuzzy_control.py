# -*- coding: utf-8 -*-
"""
# @Time    : 23/10/18 9:10 PM
# @Author  : ZHIMIN HOU
# @FileName: fuzzy_control.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
from baselines.deepq.assembly.Env_robot_control import env_search_control
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import numpy as np
import copy as cp
import skfuzzy.control as ctrl
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


class fuzzy_control(object):
    # the input is six forces and moments
    # the output is the hyperpapermeters [Kpz, kpx, kpy, krx, kry, krz]
    def __init__(self,
                 low_input=np.array([-40, -40, -40, -5, -5, -5]),
                 high_input=np.array([40, 40, 0, 5, 5, 5]),
                 low_output=np.array([0., 0., 0., 0., 0., 0.]),
                 high_output=np.array([0.015, 0.015, 0.02, 0.015, 0.015, 0.015])
                 ):

        self.low_input = low_input
        self.high_input = high_input
        self.low_output = low_output
        self.high_output = high_output
        self.num_input = 5
        self.num_output = 3
        self.num_mesh = 12

        self.sim = self.build_fuzzy_system()
        print(self.sim)

        self.unsampled = []
        for i in range(6):
            # plot the rules
            self.unsampled.append(np.linspace(self.low_input[i], self.high_input[i], 21))

    def build_fuzzy_system(self):

            # Sparse universe makes calculations faster, without sacrifice accuracy.
            # Only the critical points are included here; making it higher resolution is
            # unnecessary.
            """============================================================="""
            low_force = self.low_input
            high_force = self.high_input
            num_input = self.num_input

            fx_universe = np.linspace(low_force[0], high_force[0], num_input)
            fy_universe = np.linspace(low_force[1], high_force[1], num_input)
            fz_universe = np.linspace(low_force[2], high_force[2], num_input)

            mx_universe = np.linspace(low_force[3], low_force[3], num_input)
            my_universe = np.linspace(low_force[4], low_force[4], num_input)
            mz_universe = np.linspace(low_force[5], low_force[5], num_input)

            """Create the three fuzzy variables - two inputs, one output"""
            fx = ctrl.Antecedent(fx_universe, 'fx')
            fy = ctrl.Antecedent(fy_universe, 'fy')
            fz = ctrl.Antecedent(fz_universe, 'fz')

            mx = ctrl.Antecedent(mx_universe, 'mx')
            my = ctrl.Antecedent(my_universe, 'my')
            mz = ctrl.Antecedent(mz_universe, 'mz')

            input_names = ['nb', 'ns', 'ze', 'ps', 'pb']

            fx.automf(names=input_names)
            fy.automf(names=input_names)
            fz.automf(names=input_names)
            mx.automf(names=input_names)
            my.automf(names=input_names)
            mz.automf(names=input_names)

            """============================================================="""
            """Create the outputs"""
            kpx_universe = np.linspace(self.low_output[0], self.high_output[0], self.num_output)
            kpy_universe = np.linspace(self.low_output[1], self.high_output[1], self.num_output)
            kpz_universe = np.linspace(self.low_output[2], self.high_output[2], self.num_output)

            krx_universe = np.linspace(self.low_output[3], self.high_output[3], 3)
            kry_universe = np.linspace(self.low_output[4], self.high_output[4], 3)
            krz_universe = np.linspace(self.low_output[5], self.high_output[5], 3)

            kpx = ctrl.Consequent(kpx_universe, 'kpx')
            kpy = ctrl.Consequent(kpy_universe, 'kpy')
            kpz = ctrl.Consequent(kpz_universe, 'kpz')

            print(kpx)

            krx = ctrl.Consequent(krx_universe, 'krx')
            kry = ctrl.Consequent(kry_universe, 'kry')
            krz = ctrl.Consequent(krz_universe, 'krz')

            output_names_3 = ['nb', 'ze', 'pb']
            output_names_2 = ['nb', 'ze', 'pb']

            # Here we use the convenience `automf` to populate the fuzzy variables with
            # terms. The optional kwarg `names=` lets us specify the names of our Terms.

            kpx.automf(names=output_names_3)
            kpy.automf(names=output_names_3)
            kpz.automf(names=output_names_3)

            krx.automf(names=output_names_2)
            kry.automf(names=output_names_2)
            krz.automf(names=output_names_2)

            # define the rules for the desired force fx and my
            # ===============================================================
            rule_kpx_0 = ctrl.Rule(antecedent=((fx['nb'] & my['ze']) |
                                               (fx['nb'] & my['nb']) |
                                               (fx['nb'] & my['ns']) |
                                               (fx['pb'] & my['ze']) |
                                               (fx['pb'] & my['ps']) |
                                               (fx['pb'] & my['pb'])),
                                consequent=kpx['pb'], label='rule kpx pb')
            rule_kpx_1 = ctrl.Rule(antecedent=((fx['ns'] & my['ze']) |
                                               (fx['ns'] & my['ns']) |
                                               (fx['ns'] & my['nb']) |
                                               (fx['ps'] & my['ps']) |
                                               (fx['ps'] & my['pb']) |
                                               (fx['ps'] & my['ze'])),
                                consequent=kpx['ze'], label='rule kpx ze')
            rule_kpx_2 = ctrl.Rule(antecedent=((fx['ze'] & my['ze']) |
                                               (fx['ze'] & my['ps']) |
                                               (fx['ze'] & my['ns']) |
                                               (fx['ze'] & my['pb']) |
                                               (fx['ze'] & my['nb']) |
                                               (fx['nb'] & my['ps']) |
                                               (fx['nb'] & my['pb']) |
                                               (fx['pb'] & my['ns']) |
                                               (fx['pb'] & my['nb']) |
                                               (fx['ns'] & my['ps']) |
                                               (fx['ns'] & my['pb']) |
                                               (fx['ps'] & my['nb']) |
                                               (fx['ps'] & my['ns'])),
                                consequent=kpx['nb'], label='rule kpx nb')
            system_kpx = ctrl.ControlSystem(rules=[rule_kpx_2, rule_kpx_1, rule_kpx_0])
            sim_kpx = ctrl.ControlSystemSimulation(system_kpx, flush_after_run=self.num_mesh * self.num_mesh + 1)

            # define the rules for the desired force fy and mz
            # ===============================================================
            rule_kpy_0 = ctrl.Rule(antecedent=((fy['pb'] & mx['ze']) |
                                               (fy['nb'] & mx['nb']) |
                                               (fy['pb'] & mx['ze']) |
                                               (fy['pb'] & mx['pb'])),
                                   consequent=kpy['pb'], label='rule_kpy_pb')
            rule_kpy_1 = ctrl.Rule(antecedent=((fy['ns'] & mx['ze']) |
                                               (fy['ns'] & mx['ns']) |
                                               (fy['ns'] & mx['nb']) |
                                               (fy['ps'] & mx['ps']) |
                                               (fy['ps'] & mx['pb']) |
                                               (fy['ps'] & mx['ze']) |
                                               (fy['nb'] & mx['ns']) |
                                               (fy['pb'] & mx['ps'])),
                                   consequent=kpy['ze'], label='rule_kpy_ze')
            rule_kpy_2 = ctrl.Rule(antecedent=((fy['ze']) |
                                               (fy['nb'] & mx['ps']) |
                                               (fy['nb'] & mx['pb']) |
                                               (fy['pb'] & mx['ns']) |
                                               (fy['pb'] & mx['nb']) |
                                               (fy['ns'] & mx['ps']) |
                                               (fy['ns'] & mx['pb']) |
                                               (fy['ps'] & mx['nb']) |
                                               (fy['ps'] & mx['ns'])),
                                   consequent=kpy['nb'], label='rule_kpy_nb')
            system_kpy = ctrl.ControlSystem(rules=[rule_kpy_0, rule_kpy_1, rule_kpy_2])
            sim_kpy = ctrl.ControlSystemSimulation(system_kpy, flush_after_run=self.num_mesh * self.num_mesh + 1)

            # ===============================================================
            rule_kpz_0 = ctrl.Rule(antecedent=((fx['ze'] & fy['ze']) |
                                               (fx['ze'] & fy['ns']) |
                                               (fx['ns'] & fy['ze']) |
                                               (fx['ze'] & fy['ps']) |
                                               (fx['ps'] & fy['ze'])),
                                   consequent=kpz['pb'], label='rule_kpz_pb')
            rule_kpz_1 = ctrl.Rule(antecedent=((fx['ns'] & fy['ns']) |
                                               (fx['ps'] & fy['ps']) |
                                               (fx['ns'] & fy['ps']) |
                                               (fx['ps'] & fy['ns'])),
                                   consequent=kpz['ze'], label='rule_kpz_ze')
            rule_kpz_2 = ctrl.Rule(antecedent=((fx['nb']) |
                                               (fx['pb']) |
                                               (fy['nb']) |
                                               (fy['pb'])),
                                   consequent=kpz['nb'], label='rule_kpz_nb')
            system_kpz = ctrl.ControlSystem(rules=[rule_kpz_0, rule_kpz_1, rule_kpz_2])
            sim_kpz = ctrl.ControlSystemSimulation(system_kpz, flush_after_run=self.num_mesh * self.num_mesh + 1)

            # ===============================================================
            rule_krx_0 = ctrl.Rule(antecedent=((mx['nb'] & fy['ze']) |
                                               (mx['nb'] & fy['ns']) |
                                               (mx['pb'] & fy['ze']) |
                                               (mx['pb'] & fy['ps'])),
                                   consequent=krx['pb'], label='rule_krx_pb')
            rule_krx_1 = ctrl.Rule(antecedent=((mx['ze']) |
                                               (mx['ns']) |
                                               (mx['ps']) |
                                               (mx['nb'] & fy['nb']) |
                                               (mx['pb'] & fy['pb']) |
                                               (mx['nb'] & fy['ps']) |
                                               (mx['pb'] & fy['ns']) |
                                               (mx['nb'] & fy['pb']) |
                                               (mx['pb'] & fy['nb'])),
                                   consequent=krx['ze'], label='rule_krx_ze')
            system_krx = ctrl.ControlSystem(rules=[rule_krx_0, rule_krx_1])
            sim_krx = ctrl.ControlSystemSimulation(system_krx, flush_after_run=self.num_mesh * self.num_mesh + 1)

            # ===============================================================
            rule_kry_0 = ctrl.Rule(antecedent=((my['nb'] & fx['ze']) |
                                               (my['nb'] & fx['ns']) |
                                               (my['pb'] & fx['ze']) |
                                               (my['pb'] & fx['ps'])),
                                   consequent=kry['pb'], label='rule_kry_pb')
            rule_kry_1 = ctrl.Rule(antecedent=((my['ze']) |
                                               (my['ns']) |
                                               (my['ps']) |
                                               (my['nb'] & fx['nb']) |
                                               (my['pb'] & fx['pb']) |
                                               (my['nb'] & fx['ps']) |
                                               (my['pb'] & fx['ns']) |
                                               (my['nb'] & fx['pb']) |
                                               (my['pb'] & fx['nb'])),
                                   consequent=kry['nb'], label='rule_kry_nb')
            system_kry = ctrl.ControlSystem(rules=[rule_kry_0, rule_kry_1])
            sim_kry = ctrl.ControlSystemSimulation(system_kry, flush_after_run=self.num_mesh * self.num_mesh + 1)

            # ===============================================================
            rule_krz_0 = ctrl.Rule(antecedent=((mz['nb'] & mx['ze']) |
                                               (mz['nb'] & mx['ps']) |
                                               (mz['nb'] & mx['ps']) |
                                               (mz['pb'] & mx['ns']) |
                                               (mz['pb'] & mx['ze']) |
                                               (mz['pb'] & mx['ps'])),
                                   consequent=krz['pb'], label='rule_krz_pb')
            rule_krz_1 = ctrl.Rule(antecedent=((mz['ze']) |
                                               (mz['ns']) |
                                               (mz['ps']) |
                                               (mz['nb'] & mx['nb']) |
                                               (mz['pb'] & mx['pb']) |
                                               (mz['nb'] & mx['pb']) |
                                               (mz['pb'] & mx['nb'])),
                                   consequent=krz['ze'], label='rule_krz_ze')
            system_krz = ctrl.ControlSystem(rules=[rule_krz_0, rule_krz_1])
            sim_krz = ctrl.ControlSystemSimulation(system_krz, flush_after_run=self.num_mesh * self.num_mesh + 1)

            return [sim_kpx, sim_kpy, sim_kpz, sim_krx, sim_kry, sim_krz]

    def get_output(self, force):

        index_1 = (force[0] - self.low_input[0])/(self.high_input[0] - self.low_input[0]) * self.num_mesh
        index_2 = (force[4] - self.low_input[4])/(self.high_input[4] - self.low_input[4]) * self.num_mesh
        self.sim[0].input['fx'] = self.unsampled[index_1]
        self.sim[0].input['my'] = self.unsampled[index_2]
        self.sim[0].compute()
        kpx = self.sim[0].output['kpx']

        index_3 = (force[1] - self.low_input[1]) / (self.high_input[1] - self.low_input[1]) * self.num_mesh
        index_4 = (force[3] - self.low_input[3]) / (self.high_input[3] - self.low_input[3]) * self.num_mesh
        self.sim[1].input['fy'] = self.unsampled[index_3]
        self.sim[1].input['mx'] = self.unsampled[index_4]
        self.sim[1].compute()
        kpy = self.sim[1].output['kpy']

        index_5 = (force[0] - self.low_input[0]) / (self.high_input[0] - self.low_input[0]) * self.num_mesh
        index_6 = (force[1] - self.low_input[1]) / (self.high_input[1] - self.low_input[1]) * self.num_mesh
        self.sim[2].input['fx'] = self.unsampled[index_5]
        self.sim[2].input['fy'] = self.unsampled[index_6]
        self.sim[2].compute()
        kpz = self.sim[2].output['kpz']

        index_7 = (force[1] - self.low_input[1]) / (self.high_input[1] - self.low_input[1]) * self.num_mesh
        index_8 = (force[3] - self.low_input[3]) / (self.high_input[3] - self.low_input[3]) * self.num_mesh
        self.sim[3].input['fy'] = self.unsampled[index_7]
        self.sim[3].input['mx'] = self.unsampled[index_8]
        self.sim[3].compute()
        krx = self.sim[3].output['krx']

        index_9 = (force[0] - self.low_input[0]) / (self.high_input[0] - self.low_input[0]) * self.num_mesh
        index_10 = (force[4] - self.low_input[4]) / (self.high_input[4] - self.low_input[4]) * self.num_mesh
        self.sim[4].input['fy'] = self.unsampled[index_9]
        self.sim[4].input['mx'] = self.unsampled[index_10]
        self.sim[4].compute()
        kry = self.sim[4].output['kry']

        index_11 = (force[5] - self.low_input[5]) / (self.high_input[5] - self.low_input[5]) * self.num_mesh
        index_12 = (force[3] - self.low_input[3]) / (self.high_input[3] - self.low_input[3]) * self.num_mesh
        self.sim[3].input['mz'] = self.unsampled[index_11]
        self.sim[3].input['mx'] = self.unsampled[index_12]
        self.sim[3].compute()
        krz = self.sim[3].output['krx']
        return [kpx, kpy, kpz, krx, kry, krz]

    def plot_rules(self):
        # plt.figure(figsize=(20, 15), dpi=100)
        # plt.title('Episode Reward')
        # plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
        # plt.subplots_adjust(left=0.065, bottom=0.1, right=0.995, top=0.9, wspace=0.2, hspace=0.2)
        # plt.title("True Data")

        """kpx"""
        upsampled_x = self.unsampled[0]
        upsampled_y = self.unsampled[4]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(21):
            for j in range(21):
                self.sim[0].input['fx'] = x[i, j]
                self.sim[0].input['my'] = y[i, j]
                self.sim[0].compute()
                z[i, j] = self.sim[0].output['kpx']

        print(x)
        print(y)
        """ Plot the result in pretty 3D with alpha blending"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)

        """kpy"""
        # upsampled_x = self.unsampled[1]
        # upsampled_y = self.unsampled[3]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         self.sim[1].input['fy'] = x[i, j]
        #         self.sim[1].input['mx'] = y[i, j]
        #         self.sim[1].compute()
        #         z[i, j] = self.sim[1].output['kpy']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)

        """kpz"""
        # upsampled_x = self.unsampled[0]
        # upsampled_y = self.unsampled[1]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         self.sim[2].input['fx'] = x[i, j]
        #         self.sim[2].input['fy'] = y[i, j]
        #         self.sim[2].compute()
        #         z[i, j] = self.sim[2].output['kpz']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)
        #
        # # plot the projection in each facet
        # # cset = ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
        # # cset = ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
        # # cset = ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)
        #
        # ax.view_init(30, 200)

        """krx"""
        # upsampled_x = self.unsampled[1]
        # upsampled_y = self.unsampled[3]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         self.sim[3].input['fy'] = x[i, j]
        #         self.sim[3].input['mx'] = y[i, j]
        #         self.sim[3].compute()
        #         z[i, j] = self.sim[3].output['krx']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)

        """kry"""
        # upsampled_x = self.unsampled[0]
        # upsampled_y = self.unsampled[4]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         self.sim[4].input['fx'] = x[i, j]
        #         self.sim[4].input['my'] = y[i, j]
        #         self.sim[4].compute()
        #         z[i, j] = self.sim[4].output['kry']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)

        """krz"""
        # upsampled_x = self.unsampled[3]
        # upsampled_y = self.unsampled[5]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         self.sim[5].input['mx'] = x[i, j]
        #         self.sim[5].input['mz'] = y[i, j]
        #         self.sim[5].compute()
        #         z[i, j] = self.sim[5].output['krz']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)

        plt.show()

    def build_fuzzy_kpx(self):
        fx_universe = np.linspace(self.low_input[0], self.high_input[0], self.num_input)
        my_universe = np.linspace(self.low_input[4], self.high_input[4], self.num_input)

        fx = ctrl.Antecedent(fx_universe, 'fx')
        my = ctrl.Antecedent(my_universe, 'my')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fx.automf(names=input_names)
        my.automf(names=input_names)

        kpx_universe = np.linspace(self.low_output[0], self.high_output[0], self.num_output)
        kpx = ctrl.Consequent(kpx_universe, 'kpx')

        output_names_3 = ['nb', 'ze', 'pb']
        kpx.automf(names=output_names_3)

        rule_kpx_0 = ctrl.Rule(antecedent=((fx['nb'] & my['ze']) |
                                           (fx['nb'] & my['ns']) |
                                           (fx['pb'] & my['ze']) |
                                           (fx['pb'] & my['ps'])),
                               consequent=kpx['pb'], label='rule kpx pb')
        rule_kpx_1 = ctrl.Rule(antecedent=((fx['ns'] & my['ze']) |
                                           (fx['ns'] & my['ns']) |
                                           (fx['ns'] & my['nb']) |
                                           (fx['nb'] & my['nb']) |
                                           (fx['pb'] & my['pb']) |
                                           (fx['ps'] & my['ps']) |
                                           (fx['ps'] & my['pb']) |
                                           (fx['ps'] & my['ze'])),
                               consequent=kpx['ze'], label='rule kpx ze')
        rule_kpx_2 = ctrl.Rule(antecedent=((fx['ze'] & my['ze']) |
                                           (fx['ze'] & my['ps']) |
                                           (fx['ze'] & my['ns']) |
                                           (fx['ze'] & my['pb']) |
                                           (fx['ze'] & my['nb']) |
                                           (fx['nb'] & my['ps']) |
                                           (fx['nb'] & my['pb']) |
                                           (fx['pb'] & my['ns']) |
                                           (fx['pb'] & my['nb']) |
                                           (fx['ns'] & my['ps']) |
                                           (fx['ns'] & my['pb']) |
                                           (fx['ps'] & my['nb']) |
                                           (fx['ps'] & my['ns'])),
                               consequent=kpx['nb'], label='rule kpx nb')
        system_kpx = ctrl.ControlSystem(rules=[rule_kpx_2, rule_kpx_1, rule_kpx_0])
        sim_kpx = ctrl.ControlSystemSimulation(system_kpx, flush_after_run=self.num_mesh * self.num_mesh + 1)

        """kpx"""
        upsampled_x = self.unsampled[0]
        upsampled_y = self.unsampled[4]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(21):
            for j in range(21):
                sim_kpx.input['fx'] = x[i, j]
                sim_kpx.input['my'] = y[i, j]
                sim_kpx.compute()
                z[i, j] = sim_kpx.output['kpx']

        """ Plot the result in pretty 3D with alpha blending"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        plt.show()

    def build_fuzzy_krx(self):

        fy_universe = np.linspace(self.low_input[1], self.high_input[1], self.num_input)
        mx_universe = np.linspace(self.low_input[3], self.high_input[3], self.num_input)

        fy = ctrl.Antecedent(fy_universe, 'fy')
        mx = ctrl.Antecedent(mx_universe, 'mx')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fy.automf(names=input_names)
        mx.automf(names=input_names)

        krx_universe = np.linspace(self.low_output[3], self.high_output[3], 3)
        krx = ctrl.Consequent(krx_universe, 'krx')

        output_names_2 = ['nb', 'ze', 'pb']
        krx.automf(names=output_names_2)

        rule_krx_0 = ctrl.Rule(antecedent=((mx['nb'] & fy['ze']) |
                                           (mx['nb'] & fy['ns']) |
                                           (mx['pb'] & fy['ze']) |
                                           (mx['pb'] & fy['ps'])),
                               consequent=krx['pb'], label='rule_krx_pb')
        rule_krx_1 = ctrl.Rule(antecedent=((mx['ze']) |
                                           (mx['ns']) |
                                           (mx['ps']) |
                                           (mx['nb'] & fy['nb']) |
                                           (mx['nb'] & fy['ps']) |
                                           (mx['nb'] & fy['pb']) |
                                           (mx['pb'] & fy['pb']) |
                                           (mx['pb'] & fy['ns']) |
                                           (mx['pb'] & fy['nb'])),
                               consequent=krx['nb'], label='rule_krx_ze')
        system_krx = ctrl.ControlSystem(rules=[rule_krx_0, rule_krx_1])
        sim_krx = ctrl.ControlSystemSimulation(system_krx, flush_after_run=self.num_mesh * self.num_mesh + 1)

        upsampled_x = self.unsampled[1]
        upsampled_y = self.unsampled[3]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(21):
            for j in range(21):
                sim_krx.input['fy'] = x[i, j]
                sim_krx.input['mx'] = y[i, j]
                sim_krx.compute()
                z[i, j] = sim_krx.output['krx']

        """ Plot the result in pretty 3D with alpha blending"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        plt.show()

    def build_fuzzy_kry(self):
        fx_universe = np.linspace(self.low_input[0], self.high_input[0], self.num_input)
        my_universe = np.linspace(self.low_input[4], self.high_input[4], self.num_input)

        fx = ctrl.Antecedent(fx_universe, 'fx')
        my = ctrl.Antecedent(my_universe, 'my')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fx.automf(names=input_names)
        my.automf(names=input_names)

        kry_universe = np.linspace(self.low_output[4], self.high_output[4], 3)
        kry = ctrl.Consequent(kry_universe, 'kry')

        output_names_2 = ['nb', 'ze', 'pb']
        kry.automf(names=output_names_2)

        rule_kry_0 = ctrl.Rule(antecedent=((my['nb'] & fx['ze']) |
                                           (my['nb'] & fx['ns']) |
                                           (my['pb'] & fx['ze']) |
                                           (my['pb'] & fx['ps'])),
                               consequent=kry['pb'], label='rule_kry_pb')
        rule_kry_1 = ctrl.Rule(antecedent=((my['ze']) |
                                           (my['ns']) |
                                           (my['ps']) |
                                           (my['nb'] & fx['nb']) |
                                           (my['pb'] & fx['pb']) |
                                           (my['nb'] & fx['ps']) |
                                           (my['pb'] & fx['ns']) |
                                           (my['nb'] & fx['pb']) |
                                           (my['pb'] & fx['nb'])),
                               consequent=kry['nb'], label='rule_kry_nb')
        system_kry = ctrl.ControlSystem(rules=[rule_kry_0, rule_kry_1])
        sim_kry = ctrl.ControlSystemSimulation(system_kry, flush_after_run=self.num_mesh * self.num_mesh + 1)

        upsampled_x = self.unsampled[1]
        upsampled_y = self.unsampled[3]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(21):
            for j in range(21):
                sim_kry.input['fx'] = x[i, j]
                sim_kry.input['my'] = y[i, j]
                sim_kry.compute()
                z[i, j] = sim_kry.output['kry']

        """ Plot the result in pretty 3D with alpha blending"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        plt.show()

    def build_fuzzy_krz(self):

        mx_universe = np.linspace(self.low_input[3], self.high_input[3], self.num_input)
        mz_universe = np.linspace(self.low_input[5], self.high_input[5], self.num_input)

        mx = ctrl.Antecedent(mx_universe, 'mx')
        mz = ctrl.Antecedent(mz_universe, 'mz')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']

        mx.automf(names=input_names)
        mz.automf(names=input_names)

        krz_universe = np.linspace(self.low_output[5], self.high_output[5], 3)
        krz = ctrl.Consequent(krz_universe, 'krz')
        output_names_2 = ['nb', 'ze', 'pb']
        krz.automf(names=output_names_2)

        rule_krz_0 = ctrl.Rule(antecedent=((mz['nb'] & mx['ze']) |
                                           (mz['nb'] & mx['ps']) |
                                           (mz['nb'] & mx['ps']) |
                                           (mz['pb'] & mx['ns']) |
                                           (mz['pb'] & mx['ze']) |
                                           (mz['pb'] & mx['ps'])),
                               consequent=krz['pb'], label='rule_krz_pb')
        rule_krz_1 = ctrl.Rule(antecedent=((mz['ze']) |
                                           (mz['ns']) |
                                           (mz['ps']) |
                                           (mz['nb'] & mx['nb']) |
                                           (mz['pb'] & mx['pb']) |
                                           (mz['nb'] & mx['pb']) |
                                           (mz['pb'] & mx['nb'])),
                               consequent=krz['nb'], label='rule_krz_nb')
        system_krz = ctrl.ControlSystem(rules=[rule_krz_0, rule_krz_1])
        sim_krz = ctrl.ControlSystemSimulation(system_krz, flush_after_run=self.num_mesh * self.num_mesh + 1)

        upsampled_x = self.unsampled[3]
        upsampled_y = self.unsampled[5]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)

        for i in range(21):
            for j in range(21):
                sim_krz.input['mx'] = x[i, j]
                sim_krz.input['mz'] = y[i, j]
                sim_krz.compute()
                z[i, j] = sim_krz.output['krz']

        """ Plot the result in pretty 3D with alpha blending"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        plt.show()


if __name__ == "__main__":
    fuzzy_system = fuzzy_control()
    fuzzy_system.plot_rules()
    # fuzzy_system.build_fuzzy_kpx()