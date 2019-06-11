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
from baselines.deepq.assembly.Connect_Finall import Robot_Control
import copy as cp
from .fuzzy_control import fuzzy_control


class env_single_insert_control(object):
    def __init__(self, fuzzy=False):

        """state and Action Parameters"""
        self.action_dim = 6
        self.state_dim = 12
        self.pull_terminal = False
        self.add_noise = True
        self.fuzzy_control = fuzzy
        self.init_state = np.zeros(self.state_dim)
        self.state = np.zeros(self.state_dim)
        self.next_state = np.zeros(self.state_dim)
        self.action = np.zeros(self.action_dim)
        self.reward = 1.

        """parameters for search phase"""
        self.kp = np.array([0.02, 0.02, 0.002])
        self.kd = np.array([0.002, 0.002, 0.0002])
        self.kr = np.array([0.015, 0.015, 0.015])
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2
        self.pull_vel = 5

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The hole in world::::Tw_h=T*Tt_p, the matrix will change after installing again"""
        self.Tw_h = self.robot_control.Tw_h
        self.Tt_p = self.robot_control.T_tt
        self.search_init_position = self.robot_control.start_pos
        self.search_init_oriteation = self.robot_control.start_euler

        """[Fx, Fy, Fz, Tx, Ty, Tz]"""
        self.ref_force_insert = [0, 0, -70, 0, 0, 0]
        self.ref_force_pull = [0., 0., 80., 0., 0., 0.]

        """The safe force::F, M"""
        self.safe_force_moment = [50, 5]
        self.safe_force_insert = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """information for action and state"""
        self.high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 5])
        self.low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -5])
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(self.low, self.high)

        """The desired force and moments"""
        self.desired_force_moment = np.array([[0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0]])

        """build a fuzzy control system"""
        self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
                                high_output=np.array([0.022, 0.022, 0.015, 0.015, 0.015, 0.015]))

    """reset the start position or choose the fixed position move little step by step"""
    """ ===========================set initial position for insertion==================== """
    def reset(self):
        self.terminal = False
        self.pull_terminal = False
        self.safe_else = True
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6

        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()
        if Position_0[2] < self.robot_control.set_pos[2]:
            exit("The pegs didn't move the init position!!!")

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MoveToolTo(self.robot_control.set_pos, self.robot_control.set_euler, 10)
        self.robot_control.MoveToolTo(self.robot_control.start_insert_pos, self.robot_control.start_insert_euler, 5)

        # E_z = np.zeros(30)
        # action = np.zeros((30, 3))
        # """Move by a little step"""
        # for i in range(30):
        #
        #     myForceVector = self.robot_control.GetFCForce()
        #     if max(abs(myForceVector[0:3])) > 5:
        #         exit("The pegs can't move for the exceed force!!!")
        #
        #     """"""
        #     Position, Euler, Tw_t = self.robot_control.GetCalibTool()
        #     print(Position)
        #
        #     Tw_p = np.dot(Tw_t, self.robot_control.Tt_p)
        #     print(self.robot_control.Tw_h[2, 3])
        #
        #     E_z[i] = self.robot_control.Tw_h[2, 3] - Tw_p[2, 3]
        #     print(E_z[i])
        #
        #     if i < 3:
        #         action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
        #         vel_low = self.Kv * abs(E_z[i])
        #     else:
        #         # action[i, :] = np.array([0., 0., action[i-1, 2] + self.Kp_z_0*(E_z[i] - E_z[i-1])])
        #         action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
        #         vel_low = min(self.Kv * abs(E_z[i]), 0.5)
        #
        #     self.robot_control.MoveToolTo(Position + action[i, :], Euler, vel_low)
        #     print(action[i, :])
        #
        #     if abs(E_z[i]) < 0.001:
        #         print("The pegs reset successfully!!!")
        #         self.init_state[0:6] = myForceVector
        #         self.init_state[6:9] = Position
        #         self.init_state[9:12] = Euler
        #         break

        myForceVector = self.robot_control.GetFCForce()
        self.init_state[0:6] = myForceVector
        self.init_state[6:9] = self.robot_control.start_insert_pos
        self.init_state[9:12] = self.robot_control.start_insert_euler

        return self.init_state

    """step"""
    def step(self, action, step_num):
        """choose one action from the different actions vector"""
        done = False
        force_desired = self.ref_force_insert
        self.reward = -0.1
        force = self.state[:6]
        state = self.state[6:]

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if self.fuzzy_control:
            self.kp = self.fc.get_output(force)[:3]
            self.kr = self.fc.get_output(force)[3:]

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

        movePosition = np.zeros(6)

        # movePosition[2] = setPosition[2]
        # if action < 2:
        #     movePosition[action] = setPosition[action]
        # else:
        #     movePosition[action + 1] = setEuler[action - 2]

        movePosition[:3] = setPosition
        movePosition[3:6] = setEuler
        movePosition = movePosition + movePosition * action

        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', force)
            self.reward = -1
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MoveToolTo(state[:3] + movePosition[0][:3], state[3:] + movePosition[0][3:], setVel)
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

    """Position Control"""
    def pos_control(self):
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

    """Force Control"""
    def force_control(self, T, force_desired, force, state, step_num):
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

        """Give the required angle along z-axis"""
        setEuler[2] = 0.

        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.safe_force_moment)

        """move robot"""
        if self.safe_or_not is False:
            exit("The force is too large!!!")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MovelineTo(T, setPosition, setEuler, setVel)
            # self.robot_control.MoveToolTo(state[:3] + setPosition, state[3:] + setEuler, setVel)
            print('setPosition', setPosition)
            print('euLer', setEuler)

        if state[2] < (self.robot_control.start_insert_pos[2] - 40):
            print("=============== The insertion phase finished!!! ==================")
            done = True

        return done

    """Get the states and limit it the range"""
    def get_state(self):

        """Get the current force and moment"""
        self.state[:6] = self.robot_control.GetFCForce()
        """Get the current position"""
        self.state[6:9], self.state[9:12], T = self.robot_control.GetCalibTool()
        return self.state, T

    """Normalization of state"""
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
        final_state = (state - self.low) / scale
        return final_state

    """Pull the peg up by constant step"""
    def pull_peg_up(self):
        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MoveToolTo(Position + np.array([0., 0., 2]), Euler, self.pull_vel)

            """finish or not"""
            if Position[2] > self.robot_control.set_pos[2]:
                print("=====================Pull up the pegs finished!!!======================")
                self.pull_terminal = True
                break

        return self.pull_terminal


class env_insert_control(object):

    def __init__(self, fuzzy=False):

        """state and Action Parameters"""
        self.action_dim = 6
        self.state_dim = 12
        self.fuzzy_control = fuzzy
        self.state = np.zeros(self.state_dim)
        self.next_state = np.zeros(self.state_dim)
        self.init_state = np.zeros(self.state_dim)
        self.action = np.zeros(self.action_dim)
        self.reward = 1.

        self.pull_terminal = False
        self.add_noise = True

        """parameters for search phase"""
        self.search_kp = np.array([0.02, 0.02, 0.002])
        self.insert_kp = np.array([0.01, 0.01, 0.005])
        self.kd = np.array([0.002, 0.002, 0.0002])
        self.kr = np.array([0.02, 0.02, 0.02])
        self.kp = self.search_kp
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2
        self.pull_vel = 5

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The hole in world::::Tw_h=T*Tt_p, the matrix will change after installing again"""
        self.Tw_h = self.robot_control.Tw_h
        self.Tt_p = self.robot_control.T_tt
        # self.search_init_position = self.robot_control.start_pos
        # self.search_init_oriteation = self.robot_control.start_euler

        """[Fx, Fy, Fz, Tx, Ty, Tz]"""
        self.ref_force_search = [0, 0, -50, 0, 0, 0]
        self.ref_force_insert = [0, 0, -70, 0, 0, 0]
        self.ref_force_pull = [0., 0., 80., 0., 0., 0.]

        """The safe force::F, M"""
        self.safe_force_moment = [70, 5]
        self.safe_force_insert = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """information for action and state"""
        self.high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 5])
        self.low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -5])
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(self.low, self.high)

        """The desired force and moments"""
        self.desired_force_moment = np.array([[0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0]])

        """build a fuzzy control system"""
        self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
                                high_output=np.array([0.022, 0.022, 0.015, 0.015, 0.015, 0.015]))

    """reset the start position or choose the fixed position move little step by step"""
    """ ===========================set initial position for insertion==================== """
    def reset(self):

        self.terminal = False
        self.safe_else = True
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6
        self.Kv = 0.5

        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()

        if self.pull_terminal:
            pass
        else:
            exit("The pegs didn't move the init position!!!")

        """init_params: constant"""
        init_position = self.robot_control.set_initial_pos
        init_euler = self.robot_control.set_initial_euler

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MovelineTo(init_position, init_euler, 20)

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

            E_z[i] = self.robot_control.set_search_pos[2] - Position[2]
            print(E_z[i])

            if i < 3:
                action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
                vel_low = self.Kv * abs(E_z[i])
            else:
                # action[i, :] = np.array([0., 0., action[i-1, 2] + self.Kp_z_0*(E_z[i] - E_z[i-1])])
                action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
                vel_low = min(self.Kv * abs(E_z[i]), 0.5)

            self.robot_control.MovelineTo(Position + action[i, :], Euler, vel_low)
            print(action[i, :])

            if abs(E_z[i]) < 0.001:
                print("The pegs reset successfully!!!")
                self.init_state[0:6] = myForceVector
                self.init_state[6:9] = Position
                self.init_state[9:12] = Euler
                break

        return self.init_state

    def step(self, action, step_num):
        """choose one action from the different actions vector"""
        done = False
        force_desired = self.ref_force_insert
        self.reward = -0.1
        force = self.state[:6]
        state = self.state[6:]

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if self.fuzzy_control:
            self.kp = self.fc.get_output(force)[:3]
            self.kr = self.fc.get_output(force)[3:]

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

        movePosition = np.zeros(6)

        # movePosition[2] = setPosition[2]
        # if action < 2:
        #     movePosition[action] = setPosition[action]
        # else:
        #     movePosition[action + 1] = setEuler[action - 2]

        movePosition[:3] = setPosition
        movePosition[3:6] = setEuler
        movePosition = movePosition + movePosition * action

        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', force)
            self.reward = -1
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MovelineTo(state[:3] + movePosition[0][:3], state[3:] + movePosition[0][3:], setVel)
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

    """Position Control"""
    def pos_control(self):
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

    """Force Control"""
    def force_control(self, force_desired, force, state, step_num):
        done = False

        print("=========================Impedance Force Controller!!!===========================")
        print('force', force)

        if state[2] < self.robot_control.target_pos[2]:
            print("=========================11111111111111111111111111!!!===========================")
            print("========================= !!!!!!!!!!Begin insertion!!!===========================")
            print("=========================！！！！！！！！！！！！！！！！！===========================")
            print("Step", step_num)
            self.kp = self.insert_kp
            force_desired = self.ref_force_insert

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
            setPosition[2] = np.clip(setPosition[2], -0.2, 0.1)
            setPosition[0] = np.clip(setPosition[0], -0.1, 0.1)
            setPosition[1] = np.clip(setPosition[1], -0.1, 0.1)
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
            exit("================ The force is too large!!! ====================")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MovelineTo(state[:3] + setPosition, state[3:] + setEuler, setVel)
            print('setPosition', setPosition)
            print('euLer', setEuler)

        if state[2] < self.robot_control.set_insert_goal_pos[2]:
            print("=============== The insert phase finished!!! ==================")
            done = True

        return setPosition, setEuler, done

    """Get the states and limit it the range"""
    def get_state(self):
        force = self.robot_control.GetFCForce()
        self.state[:6] = force

        """Get the current position"""
        position, euler, T = self.robot_control.GetCalibTool()

        self.state[6:9] = position

        self.state[9:12] = euler

        # pose = self.state[6:12]
        # s = self.sensors.astype(np.float32)
        return self.state

    """Normalization of state"""
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
        final_state = (state - self.low) / scale
        return final_state

    """Pull the peg up by constant step"""
    def pull_peg_up(self):
        while True:

            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MovelineTo(Position + np.array([0., 0., 2]), Euler, self.pull_vel)

            """finish or not"""
            if Position[2] > self.robot_control.set_initial_pos[2]:
                print("=====================Pull up the pegs finished!!!======================")
                self.pull_terminal = True
                break
        return self.pull_terminal


class env_continuous_search_control(object):

    def __init__(self, fuzzy=False):
        """state and Action Parameters"""
        self.observation_dim = 12
        self.action_dim = 6
        self.state = np.zeros(self.observation_dim)
        self.next_state = np.zeros(self.observation_dim)
        self.action = np.zeros(self.action_dim)
        self.init_state = np.zeros(self.observation_dim)
        self.reward = 1.
        self.add_noise = True
        self.pull_terminal = False
        self.fuzzy_control = fuzzy
        self.step_max = 50
        self.step_max_pos = 15
        self.max_action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The desired force and moments :: get the force"""
        self.desired_force_moment = np.array([[0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0],
                                              [0, 0, -50, 0, 0, 0]])

        """The force and moment"""
        self.max_force_moment = [50, 5]
        self.safe_force_search = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """parameters for search phase"""
        self.kp = np.array([0.025, 0.025, 0.002])
        self.kd = np.array([0.005, 0.005, 0.0002])
        self.kr = np.array([0.02, 0.02, 0.02])
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2

        """information for action and state"""
        self.high = np.array([50, 50, 0, 5, 5, 6, 542, -36, 192, 5, 5, 6])
        self.low = np.array([-50, -50, -50, -5, -5, -6, 538, -42, 188, -5, -5, -6])
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(self.low, self.high)

        """build a fuzzy control system"""
        self.fc = fuzzy_control(low_output=np.array([0.01, 0.01, 0.001, 0.01, 0.01, 0.01]),
                                high_output=np.array([0.03, 0.03, 0.003, 0.02, 0.02, 0.02]))

    def reset(self):

        self.safe_else = True
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6
        self.Kv = 0.5

        # pull peg up
        self.pull_peg_up()

        if self.pull_terminal:
            pass
        else:
            exit("The pegs didn't move the init position!!!")

        # if Position_0[2] < self.robot_control.start_pos[2]:
        #     print("++++++++++++++++++++++ The pegs need to be pull up !!! +++++++++++++++++++++++++")
        #     self.pull_peg_up()

        self.robot_control.MovelineTo(self.robot_control.set_initial_pos,
                                      self.robot_control.set_initial_euler, 5)

        """add randomness for the initial position and orietation"""
        state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0., 0., 0., 0.])

        if self.add_noise:
            initial_pos = self.robot_control.set_search_pos + state_noise[0:3]
            inital_euler = self.robot_control.set_search_euler + state_noise[3:6]
            print("add noise to the initial position")
        else:
            initial_pos = self.robot_control.set_search_pos
            inital_euler = self.robot_control.set_search_euler

        """Move to the target point quickly and align with the holes"""
        _ = self.positon_control(self.robot_control.set_search_pos)
        self.robot_control.MovelineTo(initial_pos, inital_euler, 5)

        """Get the max force and moment"""
        myForceVector = self.robot_control.GetFCForce()
        max_fm = np.array([max(abs(myForceVector[0:3])), max(abs(myForceVector[3:6]))])

        safe_or_not = all(max_fm < self.max_force_moment)
        if safe_or_not is not True:
            exit("The pegs can't move for the exceed force!!!")

        # done = self.positon_control()
        done = True

        self.state = self.get_state()
        print("++++++++++++++++++++++++++++ Reset Finished !!! +++++++++++++++++++++++++++++")
        print('initial state', self.state)

        return self.get_obs(self.state), done, self.state

    def step(self, action, step_num):
        """choose one action from the different actions vector"""
        done = False
        force_desired = self.desired_force_moment[0, :]
        self.reward = -0.1
        force = self.state[:6]
        state = self.state[6:]

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if self.fuzzy_control:
            self.kp = self.fc.get_output(force)[:3]
            self.kr = self.fc.get_output(force)[3:]
            print("Kp", self.kp)
            print("Kr", self.kr)

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

        movePosition = np.zeros(6)

        # movePosition[2] = setPosition[2]
        # if action < 2:
        #     movePosition[action] = setPosition[action]
        # else:
        #     movePosition[action + 1] = setEuler[action - 2]

        movePosition[:3] = setPosition
        movePosition[3:6] = setEuler
        movePosition = movePosition + movePosition * action

        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', force)
            self.reward = -1
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MovelineTo(state[:3] + movePosition[0][:3], state[3:] + movePosition[0][3:], setVel)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('setPosition: ', setPosition)
            print('setEuLer: ', setEuler)
            print('force', force)
            print('state', state)
            # print(movePosition)

        self.next_state = self.get_state()
        if self.state[8] < self.robot_control.set_insert_pos[2]:
            print("+++++++++++++++++++++++++++++ The Search Phase Finished!!! ++++++++++++++++++++++++++++")
            self.reward = 1 - step_num/self.step_max
            done = True

        return self.get_obs(self.next_state), self.reward, done, self.safe_or_not, movePosition[0], self.next_state

    def get_reward(self, state, action):

        reward = -0.1
        force = state[:6]
        state = state[6:]

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        safe_or_not = all(max_abs_F_M < self.max_force_moment)

        if safe_or_not is False:
            reward = -1

        if state[2] < self.robot_control.set_insert_pos[2]:
            print("+++++++++++++++++++++++++++++ The Search Phase Finished!!! ++++++++++++++++++++++++++++")
            reward = 1

        return reward

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
        final_state = (state - self.low) / scale

        return final_state

    def inverse_state(self, obs):
        # normalize the state
        scale = self.high - self.low
        state = obs * scale + self.low

        return state

    def positon_control(self, target_position):

        E_z = np.zeros(30)
        action = np.zeros((30, 3))
        """Move by a little step"""
        for i in range(30):

            myForceVector = self.robot_control.GetFCForce()

            if max(abs(myForceVector[0:3])) > 5:
                exit("The pegs can't move for the exceed force!!!")

            """"""
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            # print(Position)

            E_z[i] = target_position[2] - Position[2]
            # print(E_z[i])

            if i < 3:
                action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
                vel_low = self.Kv * abs(E_z[i])
            else:
                # action[i, :] = np.array([0., 0., action[i-1, 2] + self.Kp_z_0*(E_z[i] - E_z[i-1])])
                action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
                vel_low = min(self.Kv * abs(E_z[i]), 0.5)

            self.robot_control.MovelineTo(Position + action[i, :], Euler, vel_low)
            # print(action[i, :])

            if abs(E_z[i]) < 0.001:
                print("The pegs reset successfully!!!")
                self.init_state[0:6] = myForceVector
                self.init_state[6:9] = Position
                self.init_state[9:12] = Euler
                break

        return self.init_state

        return self.init_state

    def pull_peg_up(self):
        Vel_up = 5

        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MovelineTo(Position + np.array([0., 0., 1]), Euler, Vel_up)

            """finish or not"""
            if Position[2] > self.robot_control.set_initial_pos[2]:
                print("=====================Pull up the pegs finished!!!======================")
                self.pull_terminal = True
                break

        return self.pull_terminal


class env_search_control(object):

    def __init__(self):

        """state and Action Parameters"""
        self.observation_dim = 12
        self.action_dim = 5
        self.state = np.zeros(self.observation_dim)
        self.next_state = np.zeros(self.observation_dim)
        self.init_state = np.zeros(self.observation_dim)
        self.action = np.zeros(self.action_dim)
        self.reward = 1.
        self.add_noise = True
        self.pull_terminal = False
        self.fuzzy_control = False

        self.step_max = 50
        self.step_max_pos = 15
        self.max_action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The desired force and moments :: get the force"""
        """action = [0, 1, 2, 3, 4]"""
        self.desired_force_moment = np.array([[0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0],
                                             [0, 0, -50, 0, 0, 0]])

        """The force and moment"""
        self.max_force_moment = [50, 5]
        self.safe_force_search = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """parameters for search phase"""
        self.kp = np.array([0.025, 0.025, 0.002])
        self.kd = np.array([0.005, 0.005, 0.0002])
        self.kr = np.array([0.02, 0.02, 0.02])
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2

        """information for action and state"""
        self.high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 6])
        self.low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -6])
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high)

        """build a fuzzy control system"""
        self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
                                high_output=np.array([0.022, 0.022, 0.015, 0.015, 0.015, 0.015]))

    def reset(self):

        self.safe_else = True
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6
        self.Kv = 0.5

        # pull peg up
        self.pull_peg_up()

        if self.pull_terminal:
            pass
        else:
            exit("The pegs didn't move the init position!!!")

        # if Position_0[2] < self.robot_control.start_pos[2]:
        #     print("++++++++++++++++++++++ The pegs need to be pull up !!! +++++++++++++++++++++++++")
        #     self.pull_peg_up()
        self.robot_control.MovelineTo(self.robot_control.set_initial_pos,
                                      self.robot_control.set_initial_euler, 5)

        """add randomness for the initial position and orietation"""
        state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0., 0., 0., 0.])

        if self.add_noise:
            initial_pos = self.robot_control.set_search_pos + state_noise[0:3]
            inital_euler = self.robot_control.set_search_euler + state_noise[3:6]
            print("add noise to the initial position")
        else:
            initial_pos = self.robot_control.set_search_pos
            inital_euler = self.robot_control.set_search_euler

        """Move to the target point quickly and align with the holes"""
        _ = self.positon_control(self.robot_control.set_search_pos)
        self.robot_control.MovelineTo(initial_pos, inital_euler, 5)

        """Get the max force and moment"""
        myForceVector = self.robot_control.GetFCForce()
        max_fm = np.array([max(abs(myForceVector[0:3])), max(abs(myForceVector[3:6]))])

        safe_or_not = all(max_fm < self.max_force_moment)
        if safe_or_not is not True:
            exit("The pegs can't move for the exceed force!!!")

        # done = self.positon_control()
        done = True

        self.state = self.get_state()
        print("++++++++++++++++++++++++++++ Reset Finished !!! +++++++++++++++++++++++++++++")
        print('initial state', self.state)

        return self.get_obs(self.state), done

    def step(self, action, step_num):

        """choose one action from the different actions vector"""
        done = False
        force_desired = self.desired_force_moment[action, :]
        self.reward = -0.1
        force = self.state[:6]
        state = self.state[6:]

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if self.fuzzy_control:
            self.kp = self.fc.get_output(force)[:3]
            self.kr = self.fc.get_output(force)[3:]

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

        if action < 2:
            movePosition[action] = setPosition[action]
        else:
            movePosition[action + 1] = setEuler[action - 2]

        movePosition[2] = setPosition[2]
        # movePosition[2] = -0.1
        movePosition[3:] = setEuler

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print('steps', step_num)
        print('setPosition: ', movePosition[0:3])
        print('setEuLer: ', movePosition[3:])
        print('force', force)
        print('position', state)
        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', str(force))
            self.reward = -1
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MovelineTo(state[:3] + movePosition[:3], state[3:] + setEuler, setVel)
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print('setPosition: ', setPosition)
            # print('setEuLer: ', setEuler)
            # print('force', force)

        if state[2] < self.robot_control.set_insert_pos[2]:
            print("+++++++++++++++++++++++++++++ The Search Phase Finished!!! ++++++++++++++++++++++++++++")
            self.reward = 1 - step_num/self.step_max

            done = True

        self.next_state = self.get_state()
        return self.get_obs(self.next_state), self.reward, movePosition, done, self.safe_or_not

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
        final_state = (state - self.low)/scale
        return final_state

    def positon_control(self, target_position):

        E_z = np.zeros(30)
        action = np.zeros((30, 3))
        """Move by a little step"""
        for i in range(30):

            myForceVector = self.robot_control.GetFCForce()

            if max(abs(myForceVector[0:3])) > 5:
                exit("The pegs can't move for the exceed force!!!")

            """"""
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            # print(Position)

            E_z[i] = target_position[2] - Position[2]
            # print(E_z[i])

            if i < 3:
                action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
                vel_low = self.Kv * abs(E_z[i])
            else:
                # action[i, :] = np.array([0., 0., action[i-1, 2] + self.Kp_z_0*(E_z[i] - E_z[i-1])])
                action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
                vel_low = min(self.Kv * abs(E_z[i]), 0.5)

            self.robot_control.MovelineTo(Position + action[i, :], Euler, vel_low)
            # print(action[i, :])

            if abs(E_z[i]) < 0.001:
                print("The pegs reset successfully!!!")
                self.init_state[0:6] = myForceVector
                self.init_state[6:9] = Position
                self.init_state[9:12] = Euler
                break

        return self.init_state

        # step_num = 0
        # pos_error = np.zeros(3)
        # while True:
        #     Position, Euler, Tw_t = self.robot_control.GetCalibTool()
        #     force = self.robot_control.GetFCForce()
        #     print('Force', force)
        #
        #     """Get the current position and euler"""
        #     Tw_p = np.dot(Tw_t, self.robot_control.T_tt)
        #
        #     pos_error[2] = self.robot_control.start_pos[2] - Tw_p[2, 3] - 130
        #
        #     if step_num == 0:
        #         setPostion = self.k_former * pos_error
        #         self.former_pos_error = pos_error
        #         self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
        #     elif step_num == 1:
        #         setPostion = self.k_former * pos_error
        #         self.last_pos_error = pos_error
        #         self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
        #     else:
        #         setPostion = self.k_former * pos_error
        #         # setPostion = self.k_later * (pos_error - self.last_pos_error)
        #         # self.k_later * (pos_error - 2 * self.last_pos_error + self.former_pos_error)
        #         self.former_pos_error = self.last_pos_error
        #         self.last_pos_error = pos_error
        #         self.robot_control.MoveToolTo(Position + setPostion, Euler, 0.5)
        #
        #     max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        #     safe_or_not = any(max_abs_F_M > self.safe_force_search)
        #
        #     if safe_or_not:
        #         print("Position Control finished!!!")
        #         return True
        #
        #     if step_num > self.step_max_pos:
        #         print("Position Control failed!!!")
        #         return True
        #     step_num += 1

    def pull_peg_up(self):

        Vel_up = 5
        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MovelineTo(Position + np.array([0., 0., 1]), Euler, Vel_up)

            """finish or not"""
            if Position[2] > self.robot_control.set_initial_pos[2]:
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


