import numpy as np
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from src.MNGM2 import MNGM2

class UKF:

    def __init__(self, n, m):

        self.n = n
        self.m = m

        # UKF params
        self.kappa = 0.0
        self.alfa = 0.001
        self.beta = 2.0
        self.lambda_ = (self.n + self.kappa) * self.alfa * self.alfa - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        self.W0m = self.lambda_ / (self.n + self.lambda_)
        self.W0c = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alfa * self.alfa + self.beta)
        self.W = 1.0 / (2.0 * (self.n + self.lambda_))

        # all vectors used in the UKF process
        self.x_apriori = np.zeros((self.n,), dtype=float)
        self.x_aposteriori = np.zeros((self.n,), dtype=float)
        self.x_P = np.zeros((self.n,), dtype=float)
        self.y_P = np.zeros((self.m,), dtype=float)
        self.in_ = np.zeros((self.m,), dtype=float)
        self.y = np.zeros((self.m,), dtype=float)

        # covarince matrices used in the process

        self.P_apriori = np.zeros((self.n, self.n), dtype=float)
        self.P_aprioriP = np.zeros((self.n, self.n), dtype=float)

        self.P_aposteriori = np.zeros((self.n, self.n), dtype=float)

        # square root product of a given covariances s
        self.sP_apriori = np.zeros((self.n, self.n), dtype=float)
        self.sP_aposteriori = np.zeros((self.n, self.n), dtype=float)

        # clear sigma points
        self.y_sigma = np.zeros((self.m, (2 * self.n + 1)), dtype=float)
        self.x_sigma = np.zeros((self.n, (2 * self.n + 1)), dtype=float)

        # sigma points after passing through the function f/h
        self.x_sigma_f = np.zeros((self.n, (2 * self.n + 1)), dtype=float)

        # cross covariances
        self.P_xy = np.zeros((self.n, self.m), dtype=float)
        self.P_xyP = np.zeros((self.n, self.m), dtype=float)

        self.P_y = np.zeros((self.m, self.m), dtype=float)
        self.oP_y = np.zeros((self.m, self.m), dtype=float)
        self.P_y_P = np.zeros((self.m, self.m), dtype=float)
        self.K = np.zeros((self.n, self.m), dtype=float)
        self.K_0 = np.zeros((self.n, self.m), dtype=float)
        self.K_UKF_T = np.zeros((self.m, self.n), dtype=float)

        self.Q = np.zeros((self.n, self.n), dtype=float)
        self.R = np.zeros((self.m, self.m), dtype=float)

        self.Rs = 0
        self.Qs = 0

        self.mngm = 0



    def resetUKF(self, _Q, _R, x_0):
        # Q - filter process noise covraiance
        # R - measurement noise covariance,
        # P - init covariance noise

        self.mngm = MNGM2(self.n, x_0)
        # init of all vectors and matrices where the first dim := n
        self.y = np.zeros((self.m,))
        self.y_P = np.zeros((self.m,))
        self.P_y = np.zeros((self.m, self.m))
        self.P_y_P = np.zeros((self.m, self.m))
        self.P_xy = np.zeros((self.n, self.m))
        self.P_xyP = np.zeros((self.n, self.m))

        self.K = np.zeros((self.n, self.m))
        self.K_0 = np.zeros((self.n, self.m))
        self.K_UKF_T = np.zeros((self.m, self.n))
        self.y_sigma = np.zeros((self.m, (2 * self.n + 1)))
        self.x_sigma = np.zeros((self.n, (2 * self.n + 1)))
        self.x_sigma_f = np.zeros((self.n, (2 * self.n + 1)))

        self.P_apriori = np.zeros((self.n, self.n))
        self.P_aprioriP = np.zeros((self.n, self.n))
        self.P_aposteriori = np.zeros((self.n, self.n))

        self.x_apriori = x_0[:, 0]
        self.x_aposteriori = x_0[:, 0]
        self.x_P = np.zeros((self.n,))

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

    def sigma_points(self, vect_X, matrix_S):
        # vect_X - state vector
        # sigma points are drawn from P
        self.x_sigma[:, 0] = vect_X  # the first column

        for k in range(1, self.n+1):
            self.x_sigma[:, k] = vect_X + self.gamma * matrix_S[:, k - 1]
            self.x_sigma[:, self.n + k] = vect_X - self.gamma * matrix_S[:, k - 1]

    def y_UKF_calc(self):
        # finding the y = h(x, ...)
        # the input is x_sigma, which is using h(...) then we find y_sigma_UKF from which we get to the y_UKF
        for k in range(2 * self.n + 1):
            #for i in range(self.m):
            xi = self.x_sigma_f[:, k]
            self.y_sigma[:, k] = self.mngm.output(xi)

        # y_UKF
        self.y= self.W0m * self.y_sigma[:, 0]

        for k in range(1, 2 * self.n + 1):
            self.y = self.y + self.W * self.y_sigma[:, k]

    def state(self, w):
        # w - input vector data,
        for j in range(2 * self.n + 1):
            xp = self.x_sigma[:, j]
            self.x_sigma_f[:, j] = self.mngm.state(w, xp)

    def squareRoot(self, in_):
        out_ = scipy.linalg.cholesky(in_, lower=False)
        return out_

    def timeUpdate(self, w):

        self.sP_aposteriori = self.squareRoot(self.P_aposteriori)

        self.sigma_points(self.x_aposteriori, self.sP_aposteriori)

        self.state(w)

        # apriori state:
        self.x_apriori= self.W0m * self.x_sigma_f[:, 0]
        for k in range(1, 2 * self.n + 1):
            self.x_apriori = self.x_apriori + self.W * self.x_sigma_f[:, k]


        #apriori covariance matrix:
        self.P_apriori = np.zeros((self.n, self.n))

        for k in range(2 * self.n + 1):
            #for i in range(self.n):
            self.x_P = self.x_sigma_f[:, k]

            self.x_P = self.x_P - self.x_apriori
            self.P_aprioriP = np.dot(np.expand_dims(self.x_P, axis=1), np.transpose(np.expand_dims(self.x_P, axis=1)))

            if k == 0:
                self.P_aprioriP = np.dot(self.W0c, self.P_aprioriP)
            else:
                self.P_aprioriP = np.dot(self.W, self.P_aprioriP)
            self.P_apriori = self.P_apriori + self.P_aprioriP

        self.P_apriori = self.P_apriori + self.Q

        #new sigma points can be computed here but it is not necessery
        #self.sP_apriori = self.squareRoot(self.P_apriori)
        #self.sigma_points(self.x_apriori, self.sP_apriori)

        self.y_UKF_calc()


    def measurementUpdate(self, z):
        # cov matrix oytpu/output
        self.P_y = np.zeros((self.m, self.m))

        for k in range(2 * self.n + 1):
            #for i in range(self.n):
            self.y_P = self.y_sigma[:, k]

            self.y_P = self.y_P - self.y
            self.P_y_P = np.dot(np.expand_dims(self.y_P, axis=1), np.transpose(np.expand_dims(self.y_P, axis=1)))

            if k == 0:
                self.P_y_P = np.dot(self.W0c, self.P_y_P)
            else:
                self.P_y_P = np.dot(self.W, self.P_y_P)
            self.P_y = self.P_y + self.P_y_P

        self.P_y = self.P_y + self.R


        # cross cov matrix input/output:
        self.P_xy = np.zeros((self.n, self.m))

        for k in range(2 * self.n + 1):
            self.x_P = self.x_sigma_f[:, k]
            self.y_P = self.y_sigma[:, k]

            self.x_P = self.x_P - self.x_apriori
            self.y_P = self.y_P - self.y
            self.P_xyP = np.dot(np.expand_dims(self.x_P, axis=1), np.transpose(np.expand_dims(self.y_P, axis=1)))

            if k == 0:
                self.P_xyP = np.dot(self.W0c, self.P_xyP)
            else:
                self.P_xyP = np.dot(self.W, self.P_xyP)
            self.P_xy = self.P_xy + self.P_xyP

        # kalman gain:
        self.K = np.dot(self.P_xy, np.linalg.inv(self.P_y))

        # aposteriori state:
        self.y_P = z - self.y
        self.x_aposteriori = self.x_apriori + np.dot(self.K, self.y_P)

        # cov aposteriori:
        self.P_aposteriori = self.P_apriori - np.dot(np.dot(self.K, self.P_y), np.transpose(self.K))

