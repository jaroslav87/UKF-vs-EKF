import numpy as np
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from src.MNGM2 import MNGM2

class EKF:

    def __init__(self, n, m):

        self.n = n
        self.m = m

        # all vectors used in the UKF process
        self.x_apriori = np.zeros((self.n,), dtype=float)
        self.x_aposteriori = np.zeros((self.n,), dtype=float)
        self.y_P = np.zeros((self.m,), dtype=float)
        self.y = np.zeros((self.m,), dtype=float)

        # covarince matrices used in the process

        self.F = np.zeros((self.n, self.n), dtype=float)
        self.H = np.zeros((self.m, self.n), dtype=float)

        self.P_apriori = np.zeros((self.n, self.n), dtype=float)
        self.P_aposteriori = np.zeros((self.n, self.n), dtype=float)

        self.P_y = np.zeros((self.m, self.m), dtype=float)
        self.K = np.zeros((self.n, self.m), dtype=float)

        self.Q = np.zeros((self.n, self.n), dtype=float)
        self.R = np.zeros((self.m, self.m), dtype=float)

        self.I = np.identity(2)

        self.mngm = 0



    def resetEKF(self, _Q, _R, x_0):
        # Q - filter process noise covraiance
        # R - measurement noise covariance,
        # P - init covariance noise

        self.mngm = MNGM2(self.n, x_0)
        # init of all vectors and matrices where the first dim := n

        self.y_P = np.zeros((self.m,))
        self.y = np.zeros((self.m,))

        self.P_y = np.zeros((self.m, self.m))
        # init of all vectors and matrices where the first dim := n_UKF
        self.x_apriori = x_0[:, 0]
        self.x_aposteriori = x_0[:, 0]

        self.K = np.zeros((self.n, self.m))

        self.P_apriori = np.zeros((self.n, self.n))
        self.P_aposteriori = np.zeros((self.n, self.n))

        for i in range(0, self.n):
            self.P_apriori[i, i] = _Q
            self.P_aposteriori[i, i] = _Q

        self.setCovariances(_Q, _R)

    def setCovariances(self, _Q, _R):

        self.Q = np.zeros((self.n, self.n))
        self.R = np.zeros((self.m, self.m))

        for i in range(self.n):
            self.Q[i, i] = _Q

        for i in range(self.m):
            self.R[i, i] = _R

    def timeUpdate(self, w):

        self.x_apriori = self.mngm.state(w, self.x_aposteriori)

        #apriori covariance matrix:
        self.F = self.mngm.f_jacob(self.x_apriori)
        self.P_apriori = np.matmul(self.F, np.matmul(self.P_aposteriori, np.transpose(self.F))) + self.Q


    def measurementUpdate(self, z):

        #vector residuum
        self.y = self.mngm.output(self.x_apriori)
        self.y_P = z - self.y

        #jacobian of the output
        self.H = self.mngm.h_jacob(self.x_apriori)

        # output covariance
        self.P_y = np.matmul(self.H, np.matmul(self.P_apriori, np.transpose(self.H))) + self.R

        # kalman gain:
        self.K = np.matmul(np.matmul(self.P_apriori, np.transpose(self.H)), np.linalg.inv(self.P_y))

        # aposteriori state:
        self.x_aposteriori = self.x_apriori + np.matmul(self.K, self.y_P)

        # cov aposteriori:
        self.P_aposteriori = np.matmul((self.I - np.matmul(self.K, self.H)), self.P_apriori)
