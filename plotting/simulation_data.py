import numpy as np
import matplotlib.pyplot as plt


def plot_scenario(data, order):
    plt.figure()
    plt.plot([data.measurements[i].z[0] for i in range(len(data.measurements))],
             [data.measurements[i].z[1] for i in range(len(data.measurements))],
             '.g', alpha=0.1, label='measurement')
    plt.plot(data.predictions[:, 0], data.predictions[:, order + 1], '-b', label='Prediction', alpha=1)
    plt.plot(data.states[:, 0], data.states[:, order + 1], '-r', label='True')
    plt.grid(True)
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')
    plt.ticklabel_format(axis='both', style='sci', scilimits=np.array([0, 0]))
    plt.title('Tracking scenario')
    plt.legend()
    plt.show()


def plot_prediction_error(data, t, order):
    pred_error = np.linalg.norm(data.states[:, [0, order + 1]] - data.predictions[:, [0, order + 1]], axis=1)
    plt.figure()
    plt.plot(t, pred_error)
    plt.xlabel('Time [s]')
    plt.ylabel('$|| x - \hat{x} ||_2$')
    plt.title('Position prediction error')
    plt.grid(True)
    plt.show()


def plot_snr(data):
    plt.figure()
    plt.plot(10*np.log10([data.measurements[idx].SNR for idx in range(len(data.measurements))]), 'o')
    plt.title('Measurement signal-to-noise ratio')
    plt.ylabel('SNR [dB]')
    plt.xlabel('Measurement')
    plt.grid(True)


def plot_tracking_load(data, t, dt):
    plt.figure()
    load = np.array([data.measurements[idx].n_dwells for idx in range(len(data.measurements))]) * dt*0.01 / \
           (data.revisit_intervals*dt)

    plt.plot(t[data.update_indexes], load, 'to')
    plt.ylabel('Load')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.title('Tracking load')


def plot_angular_error(data):
    plt.figure()
    plt.plot([data.measurements[i].angular_error/np.pi*180 for i in range(len(data.measurements))])


def plot_update_policy(data, t, dt):
    plt.figure()

    plt.plot(t[data.update_indexes], data.revisit_intervals*dt, '-')
    plt.xlabel('Time [s]')
    plt.ylabel('Revisit interval [s]')
    plt.title('Track update policy')
    plt.grid(True)
