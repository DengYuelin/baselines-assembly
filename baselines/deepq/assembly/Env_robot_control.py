# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Env_robot_control
   Description :  The class for real-world experiments to control the ABB robot,
                    which base on the basic class connect finally
   Author :       Zhimin Hou
   date：         18-1-9
-------------------------------------------------
   Change Activity:
                   18-1-9
-------------------------------------------------
"""
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import logging
from .Connect_Finall import Robot_Control
import pandas as pd
import copy as cp


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


class env_insert_control(object):
    def __init__(self):

        """state and Action Parameters"""
        self.action_dim = 6
        self.state_dim = 12
        self.terminal = False
        self.pull_terminal = False
        self.safe_else = True
        self.top_holes = 201.0

        """control the bounds of the sensors"""
        # sensor_0~3 = f x, y, z
        # sensor_3~6 = m x, y, z
        # sensor 6~9 = x, y, z
        # sensor 9~12 = rx, ry, rz
        # self.sensors = [1./100, 1.0/100, 1.0/30, 1.0, 1.0, 1.0,
        #                        1.0/0.1, 1.0/0.1, 1.0/200., 1.0/180., 1.0/0.05, 1.0/0.05]
        # self.actions_bounds = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 2.])

        """Fx, Fy, Fz, Mx, My, Mz"""
        self.sensors = np.zeros(self.state_dim)
        self.init_state = np.zeros(self.state_dim)

        """dx, dy, dz, rx, ry, rz"""
        self.actions = np.zeros(self.action_dim)

        """good actions parameters"""
        self.Kpz = 0.015  # [0.01, 0.02]
        self.Krxyz = 0.01  # [0.001]
        self.Kpxy = 0.0022  # [0.0005, 0.002]
        self.Kdz = 0.002
        self.Kdxy = 0.0002
        self.Vel = 5.
        self.step_max = 65
        self.Kv_fast = 2.0934

        """parameters for search phase"""
        self.kp = np.array([0.02, 0.02, 0.002])
        self.kd = np.array([0.002, 0.002, 0.0002])
        self.kr = np.array([0.015, 0.015, 0.015])
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The hole in world::::Tw_h=T*Tt_p, the matrix will change after installing again"""
        self.Tw_h = self.robot_control.Tw_h
        self.Tt_p = self.robot_control.T_tt
        self.search_init_position = self.robot_control.start_pos
        self.search_init_oriteation = self.robot_control.start_euler

        """[Fx, Fy, Fz, Tx, Ty, Tz]"""
        self.refForce = [0, 0, -50, 0, 0, 0]
        self.refForce_pull = [0., 0., 80., 0., 0., 0.]

        """The safe force::F, M"""
        self.safe_force_moment = [50, 5]
        self.safe_force_search = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """The desired force and moments"""
        self.desired_force_moment = np.array([0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0])

    """Motion step by step"""
    def step_control(self, action, step):
        """Fuzzzy reward: Including the steps and force; Only include the steps"""

        reward_methods = 'Fuzzy'

        """Get the model-basic action based on impendence control algorithm"""
        expert_actions = self.expert_actions(self.sensors[0:6])
        action = np.multiply(expert_actions, action + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        """Get the current position"""
        Position, Euler, T = self.robot_control.GetCalibTool()

        """Velocity"""
        Vel = self.Kv_fast * sum(abs(self.sensors[0:6] - self.refForce))

        """Move and rotate the pegs"""
        self.robot_control.MoveToolTo(Position + action[0:3], Euler + action[3:6], Vel)

        """Get the next force"""
        self.sensors[0:6] = self.robot_control.GetFCForce()

        """Get the next position"""
        Position_next, Euler_next, T = self.robot_control.GetCalibTool()
        self.sensors[6:9] = Position_next
        self.sensors[9:12] = Euler_next

        """Whether the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(self.sensors[0:3])), max(abs(self.sensors[3:6]))])
        self.safe_else = all(max_abs_F_M < self.Safe_Force_Moment)

        """Get the max force and moment"""
        f = max(abs(self.sensors[0:3]))
        m = max(abs(self.sensors[3:6]))
        z = self.sensors[8]
        max_depth = 40

        """reward for finished the task"""
        if z < 160:
            """change the reward"""
            Reward_final = 1.0 - step/self.step_max
            self.terminal = True
        else:
            Reward_final = 0.

        """Including three methods to design the reward function"""
        if reward_methods == 'Fuzzy':
            mfoutput, zdzoutput = frf.fuzzy_C1(m, f, 201 - z, action[5])
            Reward_process = frf.fuzzy_C2(mfoutput, zdzoutput)
        elif reward_methods == 'Time_force':
            force_reward = max(np.exp(0.02 * (f - 30)), np.exp(0.5 * (m - 1)))
            # force_reward = max(np.exp(0.01 * f), np.exp(0.3 * m)) #0.02, 0.5
            Reward_process = (-1) * (max_depth - (200 - z)) / max_depth * force_reward #[-1, 0]
        else:
            Reward_process = (-1) * (max_depth - (200 - z)) / max_depth

        Reward = Reward_final + Reward_process
        return self.sensors, Reward, self.terminal, self.safe_else

    """reset the start position or choose the fixed position move little step by step"""
    """ ===========================set initial position for insertion==================== """
    def reset(self):
        self.terminal = False
        self.pull_terminal = False
        self.safe_else = True
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6

        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()
        if Position_0[2] < 201:
            exit("The pegs didn't move the init position!!!")

        """init_params: constant"""
        init_position = np.array([6.328895870000000e+02, -44.731415000000000, 3.497448430000000e+02])
        init_euler = np.array([1.798855440000000e+02, 1.306262000000000, -0.990207000000000])

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MoveToolTo(init_position, init_euler, 20)
        self.robot_control.Align_PegHole()

        E_z = np.zeros(30)
        action = np.zeros((30, 3))
        """Move by a little step"""
        for i in range(30):

            myForceVector = self.robot_control.GetFCForce()
            if max(abs(myForceVector[0:3])) > 5:
                exit("The pegs can't move for the exceed force!!!")

            """"""
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            print(Position)

            Tw_p = np.dot(Tw_t, self.robot_control.Tt_p)
            print(self.robot_control.Tw_h[2, 3])

            E_z[i] = self.robot_control.Tw_h[2, 3] - Tw_p[2, 3]
            print(E_z[i])

            if i < 3:
                action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
                vel_low = self.Kv * abs(E_z[i])
            else:
                # action[i, :] = np.array([0., 0., action[i-1, 2] + self.Kp_z_0*(E_z[i] - E_z[i-1])])
                action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
                vel_low = min(self.Kv * abs(E_z[i]), 0.5)

            self.robot_control.MoveToolTo(Position + action[i, :], Euler, vel_low)
            print(action[i, :])

            if abs(E_z[i]) < 0.001:
                print("The pegs reset successfully!!!")
                self.init_state[0:6] = myForceVector
                self.init_state[6:9] = Position
                self.init_state[9:12] = Euler
                break

        return self.init_state

    """step predictions"""
    def step_prediction(self, action):

        """Get the current position"""
        Position, Euler, T = self.robot_control.GetCalibTool()

        """Velocity"""
        Vel = self.Kv_fast * sum(abs(self.sensors[0:6] - self.refForce))

        """Move and rotate the pegs"""
        self.robot_control.MoveToolTo(Position + action[0:3], Euler + action[3:6], Vel)

        """Get the next force"""
        self.sensors[0:6] = self.robot_control.GetFCForce()

        """Get the next position"""
        self.sensors[6:9], self.sensors[9:12], T = self.robot_control.GetCalibTool()

        """Get the max force and moment"""
        max_abs_F_M = np.array([max(abs(self.sensors[0:3])), max(abs(self.sensors[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.Safe_Force_Moment)

        """Judge whether will touch the top of holes"""
        if self.sensors[8] < self.top and self.safe_or_not:
            gamma = 0.
        else:
            gamma = 1.

        return self.sensors,  gamma, self.safe_or_not

    """===================== set initial position for search phase =================="""
    """reset the start position or choose the fixed position move little step by step"""
    def search_reset(self):
        self.pull_terminal = False
        add_noise = False

        """First the pegs need to move the initial position"""
        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()
        if Position_0[2] < self.robot_control.target_pos[2]:
            exit("The pegs can't move to the init position!!!")

        """add randomness for the initial position and orietation"""
        state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3),
                                np.random.uniform(-0.3, 0.3), 0., 0., 0.])
        if add_noise:
            initial_pos = self.robot_control.start_pos + state_noise[0:3]
            inital_euler = self.robot_control.start_euler + state_noise[3:6]
            print("add noise to the initial position")
        else:
            initial_pos = self.robot_control.start_pos
            inital_euler = self.robot_control.start_euler

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MoveToolTo(initial_pos, inital_euler, 10)

        """Get the max force and moment"""
        myForceVector = self.robot_control.GetFCForce()
        max_abs_F_M = np.array([max(abs(myForceVector[0:3])), max(abs(myForceVector[3:6]))])
        safe_or_not = all(max_abs_F_M < self.safe_force_moment)

        if safe_or_not is not True:
            exit("The pegs can't move for the exceed force!!!")

        """Get the init state with randomness"""
        self.init_state[0:6] = myForceVector
        self.init_state[6:9] = initial_pos
        self.init_state[9:12] = inital_euler
        self.pull_terminal = True
        return self.init_state, self.pull_terminal

    """=============================================================================="""
    """Position Control"""
    def pos_control(self):
        step_num = 0
        done = False
        pos_error = np.zeros(3)
        while True:
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            force = self.robot_control.GetFCForce()
            print('Force', force)

            """Get the current position and euler"""
            Tw_p = np.dot(Tw_t, self.robot_control.T_tt)

            pos_error[2] = self.robot_control.target_pos[2] - Tw_p[2, 3] - 130

            if step_num == 0:
                setPostion = self.k_former * pos_error
                self.former_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            elif step_num == 1:
                setPostion = self.k_former * pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            else:
                setPostion = self.k_former * pos_error
                # setPostion = self.k_later * (pos_error - self.last_pos_error)
                             # self.k_later * (pos_error - 2 * self.last_pos_error + self.former_pos_error)
                self.former_pos_error = self.last_pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 0.5)

            max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
            safe_or_not = any(max_abs_F_M > self.safe_force_search)

            if safe_or_not:
                done = True
                print("The pegs position control finished!!!")
                return done
            step_num += 1
            # if max(abs(force)) > 5:
            #     done = True
            #     exit("The pegs position control finished!!!")

    """Force Control"""
    def force_control(self, force_desired, force, state, step_num):
        done = False
        print("=============================================================")
        print('force', force)
        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if step_num == 0:
            setPosition = self.kp * force_error[:3]
            self.former_force_error = force_error
        elif step_num == 1:
            setPosition = self.kp * force_error[:3]
            self.last_setPosition = setPosition
            self.last_force_error = force_error
        else:
            setPosition = self.last_setPosition + self.kp * (force_error[:3] - self.last_force_error[:3]) + \
                          self.kd * (force_error[:3] - 2 * self.last_force_error[:3] + self.former_force_error[:3])
            self.last_setPosition = setPosition
            self.former_force_error = self.last_force_error
            self.last_force_error = force_error

        """Get the euler"""
        setEuler = self.kr * force_error[3:6]

        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)

        # """Get the current position"""
        # Position, Euler, T = self.robot_control.GetCalibTool()

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.safe_force_moment)

        """move robot"""
        if self.safe_or_not is False:
            exit("The force is too large!!!")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MoveToolTo(state[:3] + setPosition, state[3:] + setEuler, setVel)
            print('setPosition', setPosition)
            print('euLer', setEuler)

        if state[2] < self.robot_control.final_pos[2]:
            print("=============== The search phase finished!!! ==================")
            done = True
        return done

    """Pull the peg up"""
    def pull_search_peg(self):
        self.pull_terminal = False
        Vel_up = 5
        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MoveToolTo(Position + np.array([0., 0., 1]), Euler, Vel_up)

            """finish or not"""
            if Position[2] > self.search_init_position[2]:
                print("=====================Pull up the pegs finished!!!======================")
                self.pull_terminal = True
                break
        return self.pull_terminal

    """Get the states and limit it the range"""
    def get_state(self):
        force = self.robot_control.GetFCForce()

        """Get the current position"""
        position, euler, T = self.robot_control.GetCalibTool()

        self.sensors[6:9] = position

        self.sensors[9:12] = euler

        state = self.sensors[6:12]
        # s = self.sensors.astype(np.float32)
        return self.sensors

    """Get the fuzzy control actions"""
    def expert_actions(self, state):
        """PID Controller"""
        action = np.zeros(6)

        """The direction of Mx same with Rotx; But another is oppsite"""
        Force_error = self.refForce - state

        Force_error[0] = (-1) * Force_error[0]

        """rotate around X axis"""
        action[3] = (-1)*self.Krxyz * Force_error[3]

        """rotate around Y axis and Z axis"""
        action[4:6] = self.Krxyz * Force_error[4:6]

        """move along the X and Y axis"""
        action[0:2] = self.Kpxy * Force_error[0:2]

        """move along the Z axis"""
        action[2] = self.Kpz * Force_error[2]
        return action

    """Pull the peg up by constant step"""
    def pull_up(self):
        """Only change the action_z"""
        action = np.array([0., 0., 2., 0., 0., 0.])

        """Get the current position"""
        Position, Euler, T = self.robot_control.GetCalibTool()

        """velocities"""
        Vel_up = self.Kv_fast * sum(abs(self.sensors[0:6] - self.refForce_pull))

        """move and rotate"""
        self.robot_control.MoveToolTo(Position + action[0:3], Euler, Vel_up)

        """Get the next force"""
        self.sensors[0:6] = self.robot_control.GetFCForce()

        """Get the next position"""
        Position_next, Euler_next, T = self.robot_control.GetCalibTool()
        self.sensors[6:9] = Position_next
        self.sensors[9:12] = Euler_next

        """if the force & moment is safe for object"""
        max_abs_F_M = [max(map(abs, self.sensors[0:3])), max(map(abs, self.sensors[3:6]))]
        self.safe_else = max_abs_F_M < self.safe_force_moment

        """if finished"""
        z = self.sensors[8]
        if z > 202:
            self.pull_terminal = True
        return self.pull_terminal, self.safe_else

    """Plot the six forces"""
    def plot_force(self, Fores_moments, steps):

        # fig_force = plt.figure("Simulation")
        # plt.figure("ForceAndMoment")
        # plt.clf()
        # plt.getp(self.fig_forcemoments)
        plt.ion()
        # force_moment = np.array(Fores_moments).transpose()
        force_moment = np.array(Fores_moments)
        steps_lis = np.linspace(0, steps - 1, steps)
        ax_force = self.ax_force
        ax_force.clear()
        ax_force.plot(steps_lis, force_moment[0, :], c=COLORS[0], linewidth=2.5, label="Force_X")
        ax_force.plot(steps_lis, force_moment[1, :], c=COLORS[1], linewidth=2.5, label="Force_Y")
        ax_force.plot(steps_lis, force_moment[2, :], c=COLORS[2], linewidth=2.5, label="Force_Z")
        ax_force.plot(steps_lis, 10 * force_moment[3, :], c=COLORS[3], linewidth=2.5, label="Moment_X")
        ax_force.plot(steps_lis, 10 * force_moment[4, :], c=COLORS[4], linewidth=2.5, label="Moment_Y")
        ax_force.plot(steps_lis, 10 * force_moment[5, :], c=COLORS[5], linewidth=2.5, label="Moment_Z")

        ax_force.set_xlim(0, 40)
        ax_force.set_xlabel("Steps")
        ax_force.set_ylim(-50, 50)
        ax_force.set_ylabel("Force(N)/Moment(Ndm)")
        ax_force.legend(loc="upper right")
        ax_force.grid()
        # self.fig_forcemoments.savefig("Force_moment.jpg")
        # plt.show()
        plt.show(block=False)
        plt.pause(0.1)

    """Plot the received rewards"""
    def plot_rewards(self, Rewards, steps):

        # rewards = np.array(Rewards).transpose()
        rewards = np.array(Rewards)
        steps_lis = np.linspace(0, steps - 1, steps)

        self.ax_rewards.clear()
        self.ax_rewards.plot(steps_lis, rewards, c=COLORS[0], label='Rewards_Episode')

        self.ax_rewards.set_xlim(0, 1000)
        self.ax_rewards.set_xlabel("Episodes")
        self.ax_rewards.set_ylim(-50, 50)
        self.ax_rewards.set_ylabel("Reward")
        self.ax_rewards.legend(loc="upper right")
        self.ax_rewards.grid()

        plt.show(block=False)
        plt.pause(0.1)

    """Plot the recorded stpes"""
    def plot_steps(self, episode_steps, steps):

        # Episode_steps = np.array(episode_steps).transpose()
        Episode_steps = np.array(episode_steps)
        steps_lis = np.linspace(0, steps - 1, steps)

        self.ax_steps.plot(steps_lis, Episode_steps, c=COLORS[1], label='Steps_Episode')

        self.ax_steps.set_xlim(0, 1000)
        self.ax_steps.set_xlabel("Episodes")
        self.ax_steps.set_ylim(0, 60)
        self.ax_steps.set_ylabel("steps")
        self.ax_steps.legend(loc="upper right")
        self.ax_steps.grid()
        plt.show(block=False)
        plt.pause(0.1)

    """use to show the simulation and plot the model"""
    def plot_model(self, show_des):
        if show_des is True:
            self.PegInHolemodel.plot(self.ax_model, self.PegInHolemodel.Centers_pegs,
                                     self.PegInHolemodel.Centers_holes,
                                     self.PegInHolemodel.pegs, self.PegInHolemodel.holes)
            plt.pause(0.01)

    """Save the figure"""
    def save_figure(self, figname):

        self.fig.savefig(figname + '.jpg')

    """Read the csv"""
    def Plot_from_csv(self, csvname, figname):
        nf = pd.read_csv(csvname + ".csv", sep=',', header=None)

        fig = plt.figure(figname, figsize=(8, 8))
        ax_plot = fig.add_subplot()

        steps = np.linspace(0, len(nf) - 1, len(nf))
        plt.plot(steps, nf, c=COLORS[0], linewidth=2.5)

        ax_plot.set_xlim(0, 40)
        ax_plot.set_xlabel("Steps")
        ax_plot.set_ylim(-50, 50)
        ax_plot.set_ylabel("Force(N)/Moment(Ndm)")
        ax_plot.legend(loc="upper right")
        ax_plot.grid()

    """Calibrate the force sensor"""
    def CalibFCforce(self):
        if self.robot_control().CalibFCforce():
            init_Force = self.robot_control.GetFCForce()
            if max(abs(init_Force)) > 1:
                print("The Calibration of Force Failed!!!")
                exit()
        print("The Calibration of Force Finished!!!")
        return True


class env_search_control(object):

    def __init__(self):
        """state and Action Parameters"""
        self.observation_dim = 12
        self.action_dim = 5
        self.state = np.zeros(self.observation_dim)
        self.next_state = np.zeros(self.observation_dim)
        self.action = np.zeros(self.action_dim)
        self.reward = 1.
        self.add_noise = False
        self.pull_terminal = False
        self.step_max = 50
        self.step_max_pos = 15

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The desired force and moments :: get the force"""
        """action = [0, 1, 2, 3, 4]"""
        self.desired_force_moment = np.array([[0, 0, -40, 0, 0, 0],
                                             [0, 0, -40, 0, 0, 0],
                                             [0, 0, -40, 0, 0, 0],
                                             [0, 0, -40, 0, 0, 0],
                                             [0, 0, -40, 0, 0, 0]])

        """The force and moment"""
        self.max_force_moment = [50, 5]
        self.safe_force_search = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """parameters for search phase"""
        self.kp = np.array([0.01, 0.01, 0.0015])
        self.kd = np.array([0.005, 0.005, 0.0002])
        self.kr = np.array([0.015, 0.015, 0.015])
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2

        """information for action and state"""
        self.high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 188, 5, 5, 5])
        self.low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 192, -5, -5, -5])
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high)

    def reset(self):

        # judge whether need to pull the peg up
        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()
        if Position_0[2] < self.robot_control.start_pos[2]:
            print("++++++++++++++++++++++ The pegs need to be pull up !!! +++++++++++++++++++++++++")
            self.pull_peg_up()

        """add randomness for the initial position and orietation"""
        state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3),
                                np.random.uniform(-0.3, 0.3), 0., 0., 0.])
        if self.add_noise:
            initial_pos = self.robot_control.start_pos + state_noise[0:3]
            inital_euler = self.robot_control.start_euler + state_noise[3:6]
            print("add noise to the initial position")
        else:
            initial_pos = self.robot_control.start_pos
            inital_euler = self.robot_control.start_euler

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MoveToolTo(initial_pos, inital_euler, 10)

        """Get the max force and moment"""
        myForceVector = self.robot_control.GetFCForce()
        max_fm = np.array([max(abs(myForceVector[0:3])), max(abs(myForceVector[3:6]))])

        safe_or_not = all(max_fm < self.max_force_moment)
        if safe_or_not is not True:
            exit("The pegs can't move for the exceed force!!!")

        done = self.positon_control()

        print("++++++++++++++++++++++++++++ Reset Finished !!! +++++++++++++++++++++++++++++")
        self.state = self.get_state()
        return self.get_obs(self.state), done

    def step(self, action, step_num):

        """choose one action from the different actions vector"""
        done = False
        force_desired = self.desired_force_moment[action, :]
        self.reward = -1
        force = self.state[:6]
        state = self.state[6:]

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if step_num == 0:
            setPosition = self.kp * force_error[:3]
            self.former_force_error = force_error
        elif step_num == 1:
            setPosition = self.kp * force_error[:3]
            self.last_setPosition = setPosition
            self.last_force_error = force_error
        else:
            setPosition = self.last_setPosition + self.kp * (force_error[:3] - self.last_force_error[:3]) + \
                          self.kd * (force_error[:3] - 2 * self.last_force_error[:3] + self.former_force_error[:3])
            self.last_setPosition = setPosition
            self.former_force_error = self.last_force_error
            self.last_force_error = force_error

        """Get the euler"""
        setEuler = self.kr * force_error[3:6]

        # set the velocity of robot
        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.max_force_moment)

        movePosition = np.zeros(self.action_dim + 1)

        movePosition[2] = setPosition[2]
        if action < 2:
            movePosition[action] = setPosition[action]
        else:
            movePosition[action + 1] = setEuler[action - 2]

        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', force)
            self.reward = -10
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MoveToolTo(state[:3] + movePosition[:3], state[3:] + movePosition[3:], setVel)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('setPosition: ', setPosition)
            print('setEuLer: ', setEuler)
            print('force', force)

        if state[2] < self.robot_control.final_pos[2]:
            print("+++++++++++++++++++++++++++++ The Search Phase Finished!!! ++++++++++++++++++++++++++++")
            self.reward = 1 - step_num/self.step_max
            done = True

        self.next_state = self.get_state()
        return self.get_obs(self.next_state), self.reward, done, self.safe_or_not

    def get_state(self):
        force = self.robot_control.GetFCForce()
        position, euler, T = self.robot_control.GetCalibTool()

        self.state[:6] = force
        self.state[6:9] = position
        self.state[9:12] = euler
        return self.state

    def get_obs(self, current_state):
        state = cp.deepcopy(current_state)

        if state[9] > 0 and state[9] < 180:
            state[9] -= 180
        elif state[9] < 0 and state[9] > -180:
            state[9] += 180
        else:
            pass

        # normalize the state
        scale = self.high - self.low
        state /= scale
        return state

    def positon_control(self):
        step_num = 0
        pos_error = np.zeros(3)
        while True:
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            force = self.robot_control.GetFCForce()
            print('Force', force)

            """Get the current position and euler"""
            Tw_p = np.dot(Tw_t, self.robot_control.T_tt)

            pos_error[2] = self.robot_control.target_pos[2] - Tw_p[2, 3] - 130

            if step_num == 0:
                setPostion = self.k_former * pos_error
                self.former_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            elif step_num == 1:
                setPostion = self.k_former * pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            else:
                setPostion = self.k_former * pos_error
                # setPostion = self.k_later * (pos_error - self.last_pos_error)
                # self.k_later * (pos_error - 2 * self.last_pos_error + self.former_pos_error)
                self.former_pos_error = self.last_pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 0.5)

            max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
            safe_or_not = any(max_abs_F_M > self.safe_force_search)

            if safe_or_not:
                print("Position Control finished!!!")
                return True

            if step_num > self.step_max_pos:
                print("Position Control failed!!!")
                return True
            step_num += 1

    def pull_peg_up(self):
        Vel_up = 5
        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MoveToolTo(Position + np.array([0., 0., 1]), Euler, Vel_up)

            """finish or not"""
            if Position[2] > self.robot_control.start_pos[2]:
                print("=====================Pull up the pegs finished!!!======================")
                self.pull_terminal = True
                break
        return self.pull_terminal


class env_prediction_learn(object):
    def __init__(self):
        """state and Action Parameters"""
        self.observation_dim = 12
        self.action_dim = 6
        self.state = np.zeros(self.observation_dim)
        self.next_state = np.zeros(self.observation_dim)
        self.action = np.zeros(self.action_dim)
        self.reward = 1.
        self.add_noise = False
        self.pull_terminal = False
        self.step_max = 50
        self.step_max_pos = 15

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The desired force and moments :: get the force"""
        """action = [0, 1, 2, 3, 4]"""
        self.desired_force_moment = np.array([[0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0]])

        """The force and moment"""
        self.max_force_moment = [50, 5]
        self.safe_force_search = [5, 1]
        self.safe_force_prediction = [15, 1.5]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """parameters for search phase"""
        self.kp = np.array([0.01, 0.01, 0.0030])
        self.kd = np.array([0.005, 0.005, 0.0002])
        self.kr = np.array([0.015, 0.015, 0.015])
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2

        """information for action and state"""
        self.high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 188, 5, 5, 5])
        self.low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 192, -5, -5, -5])
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(self.low, self.high)

    def reset(self):

        # judge whether need to pull the peg up
        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()
        if Position_0[2] < self.robot_control.start_pos[2]:
            print("++++++++++++++++++++++ The pegs need to be pull up !!! +++++++++++++++++++++++++")
            self.pull_peg_up()

        """add randomness for the initial position and orietation"""
        state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3),
                                np.random.uniform(-0.3, 0.3), 0., 0., 0.])
        if self.add_noise:
            initial_pos = self.robot_control.start_pos + state_noise[0:3]
            inital_euler = self.robot_control.start_euler + state_noise[3:6]
            print("add noise to the initial position")
        else:
            initial_pos = self.robot_control.start_pos
            inital_euler = self.robot_control.start_euler

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MoveToolTo(initial_pos, inital_euler, 10)

        """Get the max force and moment"""
        myForceVector = self.robot_control.GetFCForce()
        max_fm = np.array([max(abs(myForceVector[0:3])), max(abs(myForceVector[3:6]))])

        safe_or_not = all(max_fm < self.max_force_moment)
        if safe_or_not is not True:
            exit("The pegs can't move for the exceed force!!!")

        # done = self.positon_control()

        print("++++++++++++++++++++++++++++ Reset Finished !!! +++++++++++++++++++++++++++++")
        self.state = self.get_state()
        done = True
        return self.get_obs(self.state), self.state, done

    def step(self, action):
        """move in four directions"""
        done = False

        force = self.state[:6]
        state = self.state[6:]
        force_desired = self.desired_force_moment[action, :]
        setPosition = np.zeros(6)

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        setPosition[:3] = self.kp * force_error[:3]

        """Get the euler"""
        setPosition[3:] = self.kr * force_error[3:6]
        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.safe_force_prediction)

        movePosition = np.zeros(self.action_dim)

        if action == 2:
            movePosition[action] = setPosition[action]
        else:
            movePosition[2] = setPosition[2]
            movePosition[action] = setPosition[action]

        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', force)
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MoveToolTo(state[:3] + movePosition[:3], state[3:] + movePosition[3:], setVel)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('setPosition: ', movePosition[:3])
            print('setEuLer: ', movePosition[3:])
            print('force: ', force)

        if state[2] < self.robot_control.target_pos[2]:
            print("+++++++++++++++++++++++++++++ The Search Phase Finished!!! ++++++++++++++++++++++++++++")
            done = True

        self.next_state = self.get_state()

        """Judge whether will touch the top of holes"""
        if self.state[8] < self.robot_control.target_pos[2] or self.safe_or_not is False:
            done = True
        else:
            pass

        return self.get_obs(self.next_state), self.next_state, done

    def get_state(self):
        force = self.robot_control.GetFCForce()
        position, euler, T = self.robot_control.GetCalibTool()

        self.state[:6] = force
        self.state[6:9] = position
        self.state[9:12] = euler
        return self.state

    def get_obs(self, current_state):
        state = cp.deepcopy(current_state)

        if state[9] > 0 and state[9] < 180:
            state[9] -= 180
        elif state[9] < 0 and state[9] > -180:
            state[9] += 180
        else:
            pass

        # normalize the state
        scale = self.high - self.low
        state /= scale
        return state

    def positon_control(self):
        step_num = 0
        pos_error = np.zeros(3)
        while True:
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            force = self.robot_control.GetFCForce()
            print('Force', force)

            """Get the current position and euler"""
            Tw_p = np.dot(Tw_t, self.robot_control.T_tt)

            pos_error[2] = self.robot_control.target_pos[2] - Tw_p[2, 3] - 130

            if step_num == 0:
                setPostion = self.k_former * pos_error
                self.former_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            elif step_num == 1:
                setPostion = self.k_former * pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            else:
                setPostion = self.k_former * pos_error
                # setPostion = self.k_later * (pos_error - self.last_pos_error)
                # self.k_later * (pos_error - 2 * self.last_pos_error + self.former_pos_error)
                self.former_pos_error = self.last_pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 0.5)

            max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
            safe_or_not = any(max_abs_F_M > self.safe_force_search)

            if safe_or_not:
                print("Position Control finished!!!")
                return True

            if step_num > self.step_max_pos:
                print("Position Control failed!!!")
                return True
            step_num += 1

    def pull_peg_up(self):
        Vel_up = 5
        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MoveToolTo(Position + np.array([0., 0., 1]), Euler, Vel_up)

            """finish or not"""
            if Position[2] > self.robot_control.start_pos[2]:
                print("=====================Pull up the pegs finished!!!======================")
                self.pull_terminal = True
                break
        return self.pull_terminal