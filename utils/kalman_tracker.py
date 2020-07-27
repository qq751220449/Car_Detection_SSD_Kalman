import numpy as np
from scipy.linalg import block_diag


class Kalman_Tracker():
    def __init__(self):
        # 初始化跟踪器参数
        self.id = 0         # 跟踪器ID
        self.box = []       # 存储当前跟踪目标的BBox
        self.hits = 0       # 检测目标已经检测到的次数
        self.no_losses = 0  # 跟踪目标未被检测到的次数

        # 初始化卡尔曼滤波器参数
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]

        self.x_state = []
        self.dt = 1.  # time interval

        # 状态转移矩阵
        self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # 测量转换矩阵

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])

        # 状态误差协方差矩阵-设置为各个状态误差之间相互独立
        self.L = 10.0
        self.P = np.diag(self.L * np.ones(8))

        # 系统误差变量,设定坐标位移与坐标速度之间存在相关性，坐标之间无相关性
        self.Q_comp_mat = np.array([[self.dt ** 4 / 4., self.dt ** 3 / 2.],
                                    [self.dt ** 3 / 2., self.dt ** 2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)
        # block_diag用法  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.block_diag.html

        # Initialize the measurement covariance
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)

    def update_R(self):
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)

    def predict_only(self):
        # 只进行预测,不进行修正
        self.x_state = (np.dot(self.F, self.x_state)).astype(int)     # 需要转换为坐标参数
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # 随着只进行预测,不进行修正的步数越来越多,这里的预测不确信将会越来越大
        return self.x_state

    def predict_update(self, z):
        # 预测坐标
        self.x_state = np.dot(self.F, self.x_state)  # 需要转换为坐标参数
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # 随着只进行预测,不进行修正的步数越来越多,这里的预测不确信将会越来越大

        # 更新参数
        y = z - np.dot(self.H,  self.x_state)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x_state = (self.x_state + np.dot(K, y)).astype(int)      # 计算当前时刻的估计值
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x_state


if __name__ == "__main__":
    # 创建一个跟踪器实例
    trk = Kalman_Tracker()
    trk.R_scaler = 1.0 / 16
    # 更新测量噪声协方差矩阵
    trk.update_R()
    # 初始化状态
    x_init = np.array([390, 0, 1050, 0, 513, 0, 1278, 0])
    x_init_box = [x_init[0], x_init[2], x_init[4], x_init[6]]
    # 测量
    z = np.array([399, 1022, 504, 1256])
    trk.x_state = x_init.T
    result = trk.predict_update(z.T)
    print(result)
    # 更新状态
    x_update = trk.x_state
    x_updated_box = [x_update[0], x_update[2], x_update[4], x_update[6]]

    print('The initial state is: ', x_init)
    print('The measurement is: ', z)
    print('The update state is: ', x_update)


