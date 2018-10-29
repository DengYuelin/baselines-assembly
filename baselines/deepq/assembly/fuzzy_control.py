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

    def __init__(self,
                 low_input=np.array([-40, -40, -40, -5, -5, -5]),
                 high_input=np.array([40, 40, 0, 5, 5, 5]),
                 low_output=np.array([-40, -40, -40, -5, -5, -5]),
                 high_output=np.array([-40, -40, -40, -5, -5, -5])
                 ):

        self.low_input = low_input
        self.high_input = high_input
        self.low_output = low_output
        self.high_output = high_output
        self.num_input = 5
        self.num_mesh = 21

        self.sim = self.fuzzy_control()

    def fuzzy_control(self):

        # Sparse universe makes calculations faster, without sacrifice accuracy.
        # Only the critical points are included here; making it higher resolution is
        # unnecessary.
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

        output_fx = ctrl.Consequent(fx_universe, 'output_fx')
        output_fy = ctrl.Consequent(fx_universe, 'output_fy')
        output_fz = ctrl.Consequent(fx_universe, 'output_fz')

        output_mx = ctrl.Consequent(mx_universe, 'output_fx')
        output_my = ctrl.Consequent(my_universe, 'output_my')
        output_mz = ctrl.Consequent(mz_universe, 'output_mz')

        # Here we use the convenience `automf` to populate the fuzzy variables with
        # terms. The optional kwarg `names=` lets us specify the names of our Terms.
        names = ['nb', 'ns', 'ze', 'ps', 'pb']

        fx.automf(names=names)
        fy.automf(names=names)
        fz.automf(names=names)
        mx.automf(names=names)
        my.automf(names=names)
        mz.automf(names=names)

        output_fx.automf(names=names)
        output_fy.automf(names=names)
        output_fz.automf(names=names)

        output_mx.automf(names=names)
        output_my.automf(names=names)
        output_mz.automf(names=names)

        # define the rules for the desired force fx and my
        # ===============================================================
        rule_fx_0 = ctrl.Rule(antecedent=((fx['nb'] & my['nb']) |
                                          (fx['pb'] & my['nb']) |
                                          (fx['pb'] & my['ns'])),
                            consequent=output_fx['pb'], label='rule_fx_pb')

        rule_fx_1 = ctrl.Rule(antecedent=((fx['pb'] & my['pb']) |
                                          (fx['nb'] & my['pb']) |
                                          (fx['nb'] & my['ps'])),
                            consequent=output_fx['nb'], label='rule_fx_nb')

        rule_fx_2 = ctrl.Rule(antecedent=((fx['pb'] & my['ps']) |
                                          (fx['ps'] & my['pb']) |
                                          (fx['ns'] & my['pb']) |
                                          (fx['ns'] & my['ps']) |
                                          (fx['ps'] & my['ps'])),
                            consequent=output_fx['ns'], label='rule_fx_ns')

        rule_fx_3 = ctrl.Rule(antecedent=((fx['nb'] & my['ns']) |
                                          (fx['ns'] & my['nb']) |
                                          (fx['ns'] & my['ns']) |
                                          (fx['ps'] & my['ns']) |
                                          (fx['ps'] & my['nb'])),
                            consequent=output_fx['ps'], label='rule_fx_ps')

        rule_fx_4 = ctrl.Rule(antecedent=((fx['pb'] & my['ze']) |
                                          (fx['ze'] & my['pb']) |
                                          (fx['ps'] & my['ze']) |
                                          (fx['ze'] & my['ps']) |
                                          (fx['ze'] & my['ze']) |
                                          (fx['ze'] & my['ns']) |
                                          (fx['ns'] & my['ze']) |
                                          (fx['nb'] & my['ze']) |
                                          (fx['ze'] & my['nb'])),
                            consequent=output_fx['ze'], label='rule_fx_ze')

        # define the rules for the desired force fy and mz
        # ===============================================================
        rule_fy_0 = ctrl.Rule(antecedent=((fy['nb'] & mx['nb']) |
                                          (fy['pb'] & mx['nb']) |
                                          (fy['pb'] & mx['ns'])),
                              consequent=output_fy['pb'], label='rule_fy_pb')

        rule_fy_1 = ctrl.Rule(antecedent=((fy['pb'] & mx['pb']) |
                                          (fy['nb'] & mx['pb']) |
                                          (fy['nb'] & mx['ps'])),
                              consequent=output_fy['nb'], label='rule_fy_nb')

        rule_fy_2 = ctrl.Rule(antecedent=((fy['pb'] & mx['ps']) |
                                          (fy['ps'] & mx['pb']) |
                                          (fy['ns'] & mx['pb']) |
                                          (fy['ns'] & mx['ps']) |
                                          (fy['ps'] & mx['ps'])),
                              consequent=output_fy['ns'], label='rule_fy_ns')

        rule_fy_3 = ctrl.Rule(antecedent=((fy['nb'] & mx['ns']) |
                                          (fy['ns'] & mx['nb']) |
                                          (fy['ns'] & mx['ns']) |
                                          (fy['ps'] & mx['ns']) |
                                          (fy['ps'] & mx['nb'])),
                              consequent=output_fy['ps'], label='rule_fy_ps')

        rule_fy_4 = ctrl.Rule(antecedent=((fy['pb'] & mx['ze']) |
                                          (fy['ze'] & mx['pb']) |
                                          (fy['ps'] & mx['ze']) |
                                          (fy['ze'] & mx['ps']) |
                                          (fy['ze'] & mx['ze']) |
                                          (fy['ze'] & mx['ns']) |
                                          (fy['ns'] & mx['ze']) |
                                          (fy['nb'] & mx['ze']) |
                                          (fy['ze'] & mx['nb'])),
                              consequent=output_fy['ze'], label='rule_fy_ze')

        # define the rules for the desired force fz and mx
        # ===============================================================
        rule_fz_0 = ctrl.Rule(antecedent=((fz['nb'] & mx['nb']) |
                                          (fz['pb'] & mx['nb']) |
                                          (fz['pb'] & mx['ns'])),
                              consequent=output_fz['pb'], label='rule_fz_pb')

        rule_fz_1 = ctrl.Rule(antecedent=((fz['pb'] & mx['pb']) |
                                          (fz['nb'] & mx['pb']) |
                                          (fz['nb'] & mx['ps'])),
                              consequent=output_fz['nb'], label='rule_fz_nb')

        rule_fz_2 = ctrl.Rule(antecedent=((fz['pb'] & mx['ps']) |
                                          (fz['ps'] & mx['pb']) |
                                          (fz['ns'] & mx['pb']) |
                                          (fz['ns'] & mx['ps']) |
                                          (fz['ps'] & mx['ps'])),
                              consequent=output_fz['ns'], label='rule_fz_ns')

        rule_fz_3 = ctrl.Rule(antecedent=((fz['nb'] & mx['ns']) |
                                          (fz['ns'] & mx['nb']) |
                                          (fz['ns'] & mx['ns']) |
                                          (fz['ps'] & mx['ns']) |
                                          (fz['ps'] & mx['nb'])),
                              consequent=output_fz['ps'], label='rule_fz_ps')

        rule_fz_4 = ctrl.Rule(antecedent=((fz['pb'] & mx['ze']) |
                                          (fz['ze'] & mx['pb']) |
                                          (fz['ps'] & mx['ze']) |
                                          (fz['ze'] & mx['ps']) |
                                          (fz['ze'] & mx['ze']) |
                                          (fz['ze'] & mx['ns']) |
                                          (fz['ns'] & mx['ze']) |
                                          (fz['nb'] & mx['ze']) |
                                          (fz['ze'] & mx['nb'])),
                              consequent=output_fz['ze'], label='rule_fz_ze')

        # define the rules for the desired force mx
        # ===============================================================
        rule_mx_0 = ctrl.Rule(antecedent=((fx['nb'] & my['nb']) |
                                          (fx['pb'] & my['nb']) |
                                          (fx['pb'] & my['ns'])),
                            consequent=output_mx['pb'], label='rule_mx_pb')

        rule_mx_1 = ctrl.Rule(antecedent=((fx['pb'] & my['pb']) |
                                          (fx['nb'] & my['pb']) |
                                          (fx['nb'] & my['ps'])),
                            consequent=output_mx['nb'], label='rule_mx_nb')

        rule_mx_2 = ctrl.Rule(antecedent=((fx['pb'] & my['ps']) |
                                          (fx['ps'] & my['pb']) |
                                          (fx['ns'] & my['pb']) |
                                          (fx['ns'] & my['ps']) |
                                          (fx['ps'] & my['ps'])),
                            consequent=output_mx['ns'], label='rule_mx_ze')

        # define the rules for the desired force my
        # ===============================================================
        rule_my_0 = ctrl.Rule(antecedent=((fx['nb'] & my['nb']) |
                                          (fx['pb'] & my['nb']) |
                                          (fx['pb'] & my['ns'])),
                              consequent=output_my['pb'], label='rule_my_pb')

        rule_my_1 = ctrl.Rule(antecedent=((fx['pb'] & my['pb']) |
                                          (fx['nb'] & my['pb']) |
                                          (fx['nb'] & my['ps'])),
                              consequent=output_my['nb'], label='rule_my_nb')

        rule_my_2 = ctrl.Rule(antecedent=((fx['pb'] & my['ps']) |
                                          (fx['ps'] & my['pb']) |
                                          (fx['ns'] & my['pb']) |
                                          (fx['ns'] & my['ps']) |
                                          (fx['ps'] & my['ps'])),
                              consequent=output_my['ns'], label='rule_my_ze')

        # define the rules for the desired force mz
        # ===============================================================
        rule_mz_0 = ctrl.Rule(antecedent=((mz['nb'] & fx['ze'] & fy['ze']) |
                                          (mz['nb'] & fx['ns'] & fy['ze']) |
                                          (mz['nb'] & fx['ze'] & fy['ns']) |
                                          (mz['nb'] & fx['ps'] & fy['ze']) |
                                          (mz['nb'] & fx['ze'] & fy['ps']) |
                                          (fx['pb'] & my['nb']) |
                                          (fx['pb'] & my['ns'])),
                              consequent=output_mz['pb'], label='rule_mz_pb')

        rule_mz_1 = ctrl.Rule(antecedent=((mz['pb'] & fx['ze'] & fy['ze']) |
                                          (mz['pb'] & fx['ns'] & fy['ze']) |
                                          (mz['pb'] & fx['ze'] & fy['ns']) |
                                          (mz['pb'] & fx['ps'] & fy['ze']) |
                                          (mz['pb'] & fx['ze'] & fy['ps']) |
                                          (mz[''] & fx['nb'] & fy['ps'])),
                              consequent=output_mz['nb'], label='rule_mz_nb')

        rule_mz_2 = ctrl.Rule(antecedent=((mz['ns']) |
                                          (mz['ps']) |
                                          (fx['ps'] & my['pb']) |
                                          (fx['ns'] & my['pb']) |
                                          (fx['ns'] & my['ps']) |
                                          (fx['ps'] & my['ps'])),
                              consequent=output_mz['ze'], label='rule_mz_ze')

        system_fx = ctrl.ControlSystem(rules=[rule_fx_0, rule_fx_1, rule_fx_2, rule_fx_3, rule_fx_4])
        system_fy = ctrl.ControlSystem(rules=[rule_fy_0, rule_fy_1, rule_fy_2, rule_fy_3, rule_fy_4])
        system_fz = ctrl.ControlSystem(rules=[rule_fz_0, rule_fz_1, rule_fz_2, rule_fz_3, rule_fz_4])

        system_mx = ctrl.ControlSystem(rules=[rule_mx_0, rule_mx_1, rule_mx_2])
        system_my = ctrl.ControlSystem(rules=[rule_my_0, rule_my_1, rule_my_2])
        system_mz = ctrl.ControlSystem(rules=[rule_mz_0, rule_mz_1, rule_mz_2])

        # Later we intend to run this system with a 21*21 set of inputs, so we allow
        # that many plus one unique runs before results are flushed.
        # Subsequent runs would return in 1/8 the time!
        sim_fx = ctrl.ControlSystemSimulation(system_fx, flush_after_run=self.num_mesh * self.num_mesh + 1)
        sim_fy = ctrl.ControlSystemSimulation(system_fy, flush_after_run=self.num_mesh * self.num_mesh + 1)
        sim_fz = ctrl.ControlSystemSimulation(system_fz, flush_after_run=self.num_mesh * self.num_mesh + 1)

        sim_mx = ctrl.ControlSystemSimulation(system_mx, flush_after_run=self.num_mesh * self.num_mesh + 1)
        sim_my = ctrl.ControlSystemSimulation(system_my, flush_after_run=self.num_mesh * self.num_mesh + 1)
        sim_mz = ctrl.ControlSystemSimulation(system_mz, flush_after_run=self.num_mesh * self.num_mesh + 1)

        return [sim_fx, sim_fy, sim_fz, sim_mx, sim_my, sim_mz]

    def get_output(self, input_x, input_y):
        upsampled_fx = np.linspace(self.low_input[0], self.high_input[0], self.num_mesh)
        upsampled_mx = np.linspace(self.low_input[3], self.high_input[3], self.num_mesh)

        index_1 = (input_x - self.low_input[0])/(self.high_input[0] - self.low_input[0]) * self.num_mesh
        index_2 = (input_y - self.low_input[3])/(self.high_input[3] - self.low_input[3]) * self.num_mesh

        self.sim.input['fx'] = upsampled_fx[index_1]
        self.sim.input['mx'] = upsampled_mx[index_2]
        self.sim.compute()
        output = self.sim.output['output_fx']
        return output

    def plot_rules(self):
        # plot the rules
        upsampled_x = np.linspace(-40, 40, 21)
        upsampled_y = np.linspace(-5, 5, 21)
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        for i in range(21):
            for j in range(21):
                self.sim.input['fx'] = x[i, j]
                self.sim.input['mx'] = y[i, j]
                self.sim.compute()
                z[i, j] = self.sim.output['output_fx']

        """ Plot the result in pretty 3D with alpha blending"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)

        # plot the projection in each facet
        # cset = ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
        # cset = ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
        # cset = ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)

        ax.view_init(30, 200)
        plt.show()