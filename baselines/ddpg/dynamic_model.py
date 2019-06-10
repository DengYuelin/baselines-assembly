# -*- coding: utf-8 -*-
"""
# @Time    : 08/06/19 4:38 PM
# @Author  : ZHIMIN HOU
# @FileName: dynamic_model.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from iENV import Env_PeginHole
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression

class imaginationROLLOUTS(object):

    def __init__(self):

        self.reset_localFitting()

        self.__memory_old = []
        self.__len_fitting = 5
        self.__len_old = 3000

        self.__index_old = 0

        self.__env = Env_PeginHole()
        self.__dim_s = self.__env.state_space.shape[0]
        self.__dim_a = self.__env.action_space.shape[0]

    def reset_localFitting(self):
        self.__memory_fitting = []
        self.__list_err_prior = []
        self.__list_err_linear = []
        self.__FPF = np.zeros([13, 13])
        self.__index_fitting = 0

    def store_and_fitting(self, s, a, r, s_, s_terminal):

        err_prior = s_ - self.__prior_model(s, a)
        self.__list_err_prior.append(err_prior)

        err_linear = s_ - self.__linear_model(s, a)
        self.__list_err_linear.append(err_linear)

        # assert np.all(err_linear==err_prior),(err_prior,err_linear)

        v_sas_ = s.tolist() + a.tolist() + s_.tolist()
        self.__memory_fitting.append(v_sas_)
        self.__index_fitting += 1

        if self.__index_fitting > self.__len_fitting:
            self.__list_err_prior.pop(0)
            self.__list_err_linear.pop(0)

            self.__memory_old.append(self.__memory_fitting.pop(0))  # 将fitting第一个元素弹出，存入old
            self.__index_old += 1
            if self.__index_old >= self.__len_old:
                del self.__memory_old[0]

            self.__fit_linear_model()  # 拟合局部线性模型
            self.__eval_prior_err()

    def __fit_linear_model(self):

        mean_sas_ = np.mean(self.__memory_fitting, axis=0)
        cov_sas_ = np.cov(np.array(self.__memory_fitting).T)
        cov_sas_ = cov_sas_ + np.eye(cov_sas_.shape[0]) * 1e-12

        self.__mean_1 = mean_sas_[0:self.__dim_s + self.__dim_a]
        self.__mean_2 = mean_sas_[self.__dim_s + self.__dim_a:]

        self.__cov_11 = np.mat(cov_sas_[0:self.__dim_s + self.__dim_a, 0:self.__dim_s + self.__dim_a])
        self.__cov_22 = np.mat(cov_sas_[self.__dim_s + self.__dim_a:, self.__dim_s + self.__dim_a:])
        self.__cov_12 = np.mat(cov_sas_[0:self.__dim_s + self.__dim_a, self.__dim_s + self.__dim_a:])
        self.__cov_21 = np.mat(cov_sas_[self.__dim_s + self.__dim_a:, 0:self.__dim_s + self.__dim_a])

        self.L_cholesky = np.linalg.cholesky(self.__cov_11)
        self.inv_L_cholesky = self.L_cholesky.I
        self.inv_cov_11 = self.inv_L_cholesky.T * self.inv_L_cholesky  # 为对称矩阵

        list_err_linear = []
        for sn in range(len(self.__list_err_linear)):
            list_err_linear.append(np.matmul(np.mat(self.__list_err_linear[sn]).T, np.mat(self.__list_err_linear[sn])))

        self.cov_err_linear = np.mat(np.mean(list_err_linear, axis=0))

    def __eval_prior_err(self):
        list_err_prior = []
        for sn in range(len(self.__list_err_prior)):
            list_err_prior.append(np.matmul(np.mat(self.__list_err_prior[sn]).T, np.mat(self.__list_err_prior[sn])))

        self.cov_err_prior = np.mat(np.mean(list_err_prior, axis=0))  # 对应卡尔曼滤波器中的Q矩阵

    def pred_Kalmanfilter(self, s, a):

        if self.__index_fitting < self.__len_fitting:
            return None, None, None, None

        mean_l, cov_l = self.pred_LocalLinear(s, a)
        mean_p, cov_p = self.pred_Prior(s, a)

        if mean_p is None or cov_p is None or mean_l is None or cov_l is None:
            return None, None, None, None

        assert np.all(cov_l == cov_l.T), cov_l
        assert np.all(cov_p == cov_p.T), cov_p

        cov_l = np.mat(np.diag(cov_l.diagonal().tolist()[0]))
        cov_p = np.mat(np.diag(cov_p.diagonal().tolist()[0]))

        K_kalman = cov_p * (cov_p + cov_l).I
        K_kalman = K_kalman ** 2

        # K_kalman =np.mat(np.eye(self.__dim_s)*0.5)

        # print(K_kalman)

        diff_mean = mean_l - mean_p
        mean_s_ = mean_p + np.matmul(K_kalman, diff_mean)
        # mean_s_ = (mean_p + mean_l) * 0.5

        mean_s_ = np.array(mean_s_.tolist()[0])
        # mean_s_ = np.clip(mean_s_, -1.5, 1.5)

        cov_s_ = cov_p - np.matmul(K_kalman, cov_p)

        return mean_s_, K_kalman, cov_p, cov_l

    def func_Kalmanfilter(self):

        def func(s, a):
            s_, _, _, _ = self.pred_Kalmanfilter(s, a)
            return s_

        return func

    def pred_Average(self, s, a):

        if not self.flag_ready:
            return None, None, None, None

        mean_l, _ = self.pred_LocalLinear(s, a)
        mean_p, _ = self.pred_Prior(s, a)

        K_kalman = np.mat(np.eye(self.__dim_s) * 0.5)

        diff_mean = mean_l - mean_p
        mean_s_ = mean_p + np.matmul(K_kalman, diff_mean)
        mean_s_ = np.array(mean_s_.tolist()[0])

        return mean_s_

    def pred_LocalLinear(self, s, a):

        if not self.flag_ready:
            return None, None

        mean_s_ = self.__linear_model(s, a)

        # var_cov = np.matmul(np.matmul(self.__cov_21, self.inv_cov_11), self.__cov_12)
        tm = self.inv_L_cholesky * self.__cov_12
        var_cov = tm.T * tm

        cov_s_ = self.__cov_22 - var_cov

        cov_s_ = self.cov_err_linear + np.eye(self.cov_err_linear.shape[0]) * 1e-13 * 0

        # assert np.all(self.__cov_21 == self.__cov_12.T), cov_s_
        # assert np.all(var_cov == var_cov.T), var_cov
        # assert np.all(cov_s_ == cov_s_.T), cov_s_

        return mean_s_, cov_s_

    def pred_Prior(self, s, a):

        if not self.flag_ready:
            return None, None

        mean_s_ = self.__prior_model(s, a)

        cov_s_ = self.cov_err_prior + np.eye(self.cov_err_prior.shape[0]) * 1e-13 * 0

        return mean_s_, cov_s_

    def __prior_model(self, s, a):
        s = self.__env.decode_state(s)
        a = self.__env.decode_action(a)
        s_ = np.zeros(13)
        # s_[0] = s[0] + a[0] * a[3]
        # s_[1] = s[1] + a[1] * a[4]
        # s_[2] = s[2] + a[2] * a[5]
        s_[0] = (s[3] + a[0]) * a[3]
        s_[1] = (s[4] + a[1]) * a[4]
        s_[2] = (s[5] + a[2]) * a[5]
        s_[3] = s[3] + a[0]
        s_[4] = s[4] + a[1]
        s_[5] = s[5] + a[2]
        s_[6] = s_[0] * 0.0001
        s_[7] = s_[1] * 0.0001
        s_[8] = s_[2] * 0.0001
        s_[9] = a[0] * a[3]
        s_[10] = a[1] * a[4]
        s_[11] = a[2] * a[5]
        s_[12] = s[12] - s_[7] / self.__env.depth_hole
        s_ = self.__env.code_state(s_)

        return s_

    def __linear_model(self, s, a):

        if not self.flag_ready:
            return self.__prior_model(s, a)

        x1 = s.tolist() + a.tolist()
        x1 = np.array(x1)

        s_ = self.__mean_2 + np.matmul(self.__cov_21 * self.inv_cov_11, x1 - self.__mean_1)
        s_ = np.array(s_.tolist()[0])

        return s_

    @property
    def flag_ready(self):
        return self.__index_fitting > self.__len_fitting

    @property
    def flag_jamming(self):

        if not self.flag_ready:
            return False

        list_s_ = np.array(self.__memory_fitting)[:, self.__dim_s + self.__dim_a:]
        list_process = list_s_[:, -1]

        dx_max = np.max(list_process)
        dx_min = np.min(list_process)

        dx_max = list_process[-1]
        dx_min = list_process[0]

        flag = (dx_max - dx_min < 0.001) and (dx_min > 0.6)
        flag = dx_max - dx_min < 0.001

        if flag:
            print('  【iIMAGINATION消息】检测到卡死。')

        return flag


if __name__ == '__main__':

    data = np.load('./data/second_data/train_states_fuzzy.npy')
    # print(data[0][:, 3])
    print(data[0][0][0])

    x = np.array(np.array(data[0]).reshape(-1, 4)[:, 3][:3]).flatten()
    y = np.array(data[0]).reshape(-1, 4)[:, 3][1]

    X = np.zeros((len(data[0]), 18), dtype=np.float32)
    Y = np.zeros((len(data[0]), 12), dtype=np.float32)
    for i in range(len(data[0])):
        X[i, :12] = data[0][i][0]
        X[i, 12:] = data[0][i][1]
        Y[i, :] = data[0][i][3]

    model_type = 'gp'

    if model_type == 'gp':
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        dynamic_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    elif model_type == 'linear':
        dynamic_model = LinearRegression()
    else:
        dynamic_model = LinearRegression()

    x = np.array([[1., 1.], [1.2, 2.1]])
    y = np.array([[2., 3.], [3., 4.]])

    dynamic_model.fit(X, Y)

    a = np.zeros((2, 10), dtype=np.float32)
    b = np.zeros((2, 10), dtype=np.float32)
    print(np.append(a, b, axis=0))

