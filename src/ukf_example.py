import os.path

from src.EKF import EKF
from src.MNGM2 import MNGM2
from src.UKF import UKF
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def plot_confidence_ellipse_cov(dataX, dataY, est_state, est_y, P, Py):
    n = dataX.shape[0]

    sig = np.zeros((n, 4))  ## x0 x1 y0 y1
    sig_mean = np.zeros(4)  # mean(x0) mean(x1) mean(y0) mean(y1)

    est_mean = np.mean(est_state, axis=0)
    est_y_mean = np.mean(est_y, axis=0)
    for i in range(2):
        sig[:, i] = dataX[:, i]
        sig[:, i + 2] = dataY[:, i]

        sig_mean[i] = np.mean(sig[:, i])
        sig_mean[i + 2] = np.mean(sig[:, i + 2])
    # y_mean = np.mean(x[:, 1])
    # p_aposteriori cov
    cov1 = np.zeros((2, 2))
    cov1[0, 0] = np.mean(P[:, 0])
    cov1[0, 1] = np.mean(P[:, 1])
    cov1[1, 0] = np.mean(P[:, 2])
    cov1[1, 1] = np.mean(P[:, 3])
    lambda_p, v = np.linalg.eig(cov1)
    lambda_p = np.sqrt(lambda_p)
    vvv = v[:, 0][::-1]
    aaa = np.arctan2(vvv[0], vvv[1])
    anglep = np.arctan2(*v[:, 0][::-1])

    # p_y cov
    cov1[0, 0] = np.mean(Py[:, 0])
    cov1[0, 1] = np.mean(Py[:, 1])
    cov1[1, 0] = np.mean(Py[:, 2])
    cov1[1, 1] = np.mean(Py[:, 3])
    lambda_py, v = np.linalg.eig(cov1)
    lambda_py = np.sqrt(lambda_py)
    anglepy = np.arctan2(*v[:, 0][::-1])

    angle = np.zeros(2)
    lambda_ = np.zeros(4)
    # first ellipse:
    cov1 = np.cov(sig[:, 0], sig[:, 1])
    lambda_1, v = np.linalg.eig(cov1)
    lambda_1 = np.sqrt(lambda_1)
    lambda_[0:2] = lambda_1
    angle[0] = np.arctan2(*v[:, 0][::-1])

    # second ellipse:
    cov2 = np.cov(sig[:, 2], sig[:, 3])
    lambda_2, v = np.linalg.eig(cov2)
    lambda_2 = np.sqrt(lambda_2)
    lambda_[2:4] = lambda_2
    angle[1] = np.arctan2(*v[:, 0][::-1])

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    for i, ax in enumerate(axs):  # ax in axs:
        ax.scatter(sig[:, i * 2], sig[:, i * 2 + 1], s=0.9)

        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        if i == 0:
            ellp = Ellipse(xy=(est_mean[0], est_mean[1]),
                           width=lambda_p[0] * 6, height=lambda_p[1] * 6,
                           angle=np.rad2deg(anglep), edgecolor='black', label=r'$3\sigma$')
            ellp.set(label=r'$3\sigma$')
            ax.add_artist(ellp)
            ellp.set_facecolor('none')

            ax.scatter(est_mean[0], est_mean[1], c='black', s=3)

        if i == 1:
            ellp = Ellipse(xy=(est_y_mean[0], est_y_mean[1]),
                           width=lambda_py[0] * 6, height=lambda_py[1] * 6,
                           angle=np.rad2deg(anglepy), edgecolor='black', label=r'$3\sigma$')
            ellp.set(label=r'$3\sigma$')
            ax.add_artist(ellp)
            ellp.set_facecolor('none')

            ax.scatter(est_y_mean[0], est_y_mean[1], c='black', s=3)

        ell = Ellipse(xy=(sig_mean[i * 2], sig_mean[i * 2 + 1]),
                      width=lambda_[i * 2] * 6, height=lambda_[i * 2 + 1] * 6,
                      angle=np.rad2deg(angle[i]), edgecolor='firebrick', label=r'$3\sigma$')
        ell.set(label=r'$3\sigma$')
        ax.add_artist(ell)
        ell.set_facecolor('none')
        ax.set(xlim=[-10, 10], ylim=[-15, 5], aspect='equal')
        ax.scatter(sig_mean[i * 2], sig_mean[i * 2 + 1], c='red', s=3)
        if i == 0:
            ax.set_title('cov state var')
        else:
            ax.set_title('cov output var')
    plt.show()




def plot_confidence_ellipse_cov_all(dataX, dataY, est_state_ekf, est_y_ekf, P_ekf, Py_ekf,
                                                est_state_ukf, est_y_ukf, P_ukf, Py_ukf):
    n = dataX.shape[0]

    sig = np.zeros((n, 4))  ## x0 x1 y0 y1
    sig_mean = np.zeros(4)  # mean(x0) mean(x1) mean(y0) mean(y1)

    est_mean_ekf = np.mean(est_state_ekf, axis=0)
    est_y_mean_ekf = np.mean(est_y_ekf, axis=0)
    est_mean_ukf = np.mean(est_state_ukf, axis=0)
    est_y_mean_ukf = np.mean(est_y_ukf, axis=0)

    for i in range(2):
        sig[:, i] = dataX[:, i]
        sig[:, i + 2] = dataY[:, i]

        sig_mean[i] = np.mean(sig[:, i])
        sig_mean[i + 2] = np.mean(sig[:, i + 2])

    #cov of EKF P
    cov1 = np.zeros((2, 2))
    cov1[0, 0] = np.mean(P_ekf[:, 0])
    cov1[0, 1] = np.mean(P_ekf[:, 1])
    cov1[1, 0] = np.mean(P_ekf[:, 2])
    cov1[1, 1] = np.mean(P_ekf[:, 3])
    lambda_p_ekf, v = np.linalg.eig(cov1)
    lambda_p_ekf = np.sqrt(lambda_p_ekf)
    vvv = v[:, 0][::-1]
    aaa = np.arctan2(vvv[0], vvv[1])
    angle_p_ekf = np.arctan2(*v[:, 0][::-1])

    # EKF p_y cov
    cov1[0, 0] = np.mean(Py_ekf[:, 0])
    cov1[0, 1] = np.mean(Py_ekf[:, 1])
    cov1[1, 0] = np.mean(Py_ekf[:, 2])
    cov1[1, 1] = np.mean(Py_ekf[:, 3])
    lambda_py_ekf, v = np.linalg.eig(cov1)
    lambda_py_ekf = np.sqrt(lambda_py_ekf)
    angle_py_ekf = np.arctan2(*v[:, 0][::-1])

    # cov of UKF P
    cov1 = np.zeros((2, 2))
    cov1[0, 0] = np.mean(P_ukf[:, 0])
    cov1[0, 1] = np.mean(P_ukf[:, 1])
    cov1[1, 0] = np.mean(P_ukf[:, 2])
    cov1[1, 1] = np.mean(P_ukf[:, 3])
    lambda_p_ukf, v = np.linalg.eig(cov1)
    lambda_p_ukf = np.sqrt(lambda_p_ukf)
    vvv = v[:, 0][::-1]
    aaa = np.arctan2(vvv[0], vvv[1])
    angle_p_ukf = np.arctan2(*v[:, 0][::-1])

    # UKF p_y cov
    cov1[0, 0] = np.mean(Py_ukf[:, 0])
    cov1[0, 1] = np.mean(Py_ukf[:, 1])
    cov1[1, 0] = np.mean(Py_ukf[:, 2])
    cov1[1, 1] = np.mean(Py_ukf[:, 3])
    lambda_py_ukf, v = np.linalg.eig(cov1)
    lambda_py_ukf = np.sqrt(lambda_py_ukf)
    angle_py_ukf = np.arctan2(*v[:, 0][::-1])

    #signal cov
    angle = np.zeros(2)
    lambda_ = np.zeros(4)
    # first ellipse:
    cov1 = np.cov(sig[:, 0], sig[:, 1])
    lambda_1, v = np.linalg.eig(cov1)
    lambda_1 = np.sqrt(lambda_1)
    lambda_[0:2] = lambda_1
    angle[0] = np.arctan2(*v[:, 0][::-1])

    # second ellipse:
    cov2 = np.cov(sig[:, 2], sig[:, 3])
    lambda_2, v = np.linalg.eig(cov2)
    lambda_2 = np.sqrt(lambda_2)
    lambda_[2:4] = lambda_2
    angle[1] = np.arctan2(*v[:, 0][::-1])

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    for i, ax in enumerate(axs):  # ax in axs:
        ax.scatter(sig[:, i * 2], sig[:, i * 2 + 1], s=0.9)

        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        if i == 0:
            ell_p_ekf = Ellipse(xy=(est_mean_ekf[0], est_mean_ekf[1]),
                           width=lambda_p_ekf[0] * 6, height=lambda_p_ekf[1] * 6,
                           angle=np.rad2deg(angle_p_ekf), edgecolor='blue', label=r'$3\sigma$')
            ell_p_ekf.set(label=r'$3\sigma$')
            ax.add_artist(ell_p_ekf)
            ell_p_ekf.set_facecolor('none')

            ax.scatter(est_mean_ekf[0], est_mean_ekf[1], c='blue', s=3)

            ell_p_ukf = Ellipse(xy=(est_mean_ukf[0], est_mean_ukf[1]),
                           width=lambda_p_ukf[0] * 6, height=lambda_p_ukf[1] * 6,
                           angle=np.rad2deg(angle_p_ukf), edgecolor='green', label=r'$3\sigma$')
            ell_p_ukf.set(label=r'$3\sigma$')
            ax.add_artist(ell_p_ukf)
            ell_p_ukf.set_facecolor('none')

            ax.scatter(est_mean_ukf[0], est_mean_ukf[1], c='green', s=3)

        if i == 1:
            ell_py_ekf = Ellipse(xy=(est_y_mean_ekf[0], est_y_mean_ekf[1]),
                           width=lambda_py_ekf[0] * 6, height=lambda_py_ekf[1] * 6,
                           angle=np.rad2deg(angle_py_ekf), edgecolor='blue', label=r'$3\sigma$')
            ell_py_ekf.set(label=r'$3\sigma$')
            ax.add_artist(ell_py_ekf)
            ell_py_ekf.set_facecolor('none')

            ax.scatter(est_y_mean_ekf[0], est_y_mean_ekf[1], c='blue', s=3)

            ell_py_ukf = Ellipse(xy=(est_y_mean_ukf[0], est_y_mean_ukf[1]),
                           width=lambda_py_ukf[0] * 6, height=lambda_py_ukf[1] * 6,
                           angle=np.rad2deg(angle_py_ekf), edgecolor='green', label=r'$3\sigma$')
            ell_py_ukf.set(label=r'$3\sigma$')
            ax.add_artist(ell_py_ukf)
            ell_py_ukf.set_facecolor('none')

            ax.scatter(est_y_mean_ukf[0], est_y_mean_ukf[1], c='green', s=3)

        ell = Ellipse(xy=(sig_mean[i * 2], sig_mean[i * 2 + 1]),
                      width=lambda_[i * 2] * 6, height=lambda_[i * 2 + 1] * 6,
                      angle=np.rad2deg(angle[i]), edgecolor='firebrick', label=r'$3\sigma$')
        ell.set(label=r'$3\sigma$')
        ax.add_artist(ell)
        ell.set_facecolor('none')
        #ax.set(xlim=[-10, 10], ylim=[-15, 5], aspect='equal')
        ax.scatter(sig_mean[i * 2], sig_mean[i * 2 + 1], c='red', s=3)
        if i == 0:
            ax.set_title('cov state var')
        else:
            ax.set_title('cov output var')
    plt.show()


def estimateState():
    n = 2  # size of the state vector
    m = 2  # size of the output vector

    # initial x value
    x_0 = np.zeros((n, 1))
    x_0[0, 0] = 1.1
    x_0[1, 0] = 2.1
    # added offset in the model to capture
    mngm = MNGM2(5000, x_0)
    mngm.generate_data()

    ukf = UKF(n, m)
    ekf = EKF(n, m)

    # generated data:
    dataX = mngm.x
    dataY = mngm.y

    size_n = dataX.shape[0]

    ukf.resetUKF(3.0, 1.0, x_0)
    ekf.resetEKF(3.0, 1.0, x_0)

    err_ukf = 0
    err_ekf = 0

    est_state_ukf = np.zeros((size_n, n))
    est_y_ukf = np.zeros((size_n, m))
    est_P_ukf = np.zeros((size_n, n * 2))
    est_Py_ukf = np.zeros((size_n, n * 2))

    est_state_ekf = np.zeros((size_n, n))
    est_y_ekf = np.zeros((size_n, m))
    est_P_ekf = np.zeros((size_n, n * 2))
    est_Py_ekf = np.zeros((size_n, n * 2))

    # estimation loop
    for i in range(size_n):

        timeUpdateInput = i
        measurementUpdateInput = dataY[i, :]

        # recursively go through time update and measurement correction

        ekf.timeUpdate(timeUpdateInput)
        ekf.measurementUpdate(measurementUpdateInput)

        ukf.timeUpdate(timeUpdateInput)
        ukf.measurementUpdate(measurementUpdateInput)

        err_ukf = err_ukf + np.sum((ukf.x_aposteriori - dataX[i, :]) ** 2)
        err_ekf = err_ekf + np.sum((ekf.x_aposteriori - dataX[i, :]) ** 2)


        est_state_ukf[i, :] = ukf.x_aposteriori

        est_P_ukf[i, :] = ukf.P_aposteriori.flatten()
        est_Py_ukf[i, :] = ukf.P_y.flatten()

        est_y_ukf[i, :] = ukf.y

        est_state_ekf[i, :] = ekf.x_aposteriori
        est_P_ekf[i, :] = ekf.P_aposteriori.flatten()

        est_Py_ekf[i, :] = ekf.P_y.flatten()

        est_y_ekf[i, :] = ekf.y

    #err_ukf = err_ukf /size_n
    #err_ekf = err_ekf /size_n

    print("total error ukf:", err_ukf)
    print("total error ekf:", err_ekf)



    plt.plot(dataX[:, 0], 'g', label='x_1 original')  # X from the orginal ungm
    plt.plot(dataX[:, 1], 'b', label='x_2 original')  # X from the orginal ungm
    plt.plot(est_state_ukf[:, 0], 'r--', label='x_1 ukf') #estimated X
    plt.plot(est_state_ukf[:, 1], 'k--', label='x_2 ukf')  # estimated X
    plt.plot(est_state_ekf[:, 0], 'b--', label='x_1 ekf')  # estimated X
    plt.plot(est_state_ekf[:, 1], 'c--', label='x_2 ekf')  # estimated X
    plt.legend(loc='upper right')
    plt.show()

    plot_confidence_ellipse_cov_all(dataX, dataY,
                                est_state_ekf, est_y_ekf, est_P_ekf, est_Py_ekf,
                                est_state_ukf, est_y_ukf, est_P_ukf, est_Py_ukf)

estimateState()
