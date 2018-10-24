import numpy as np
import matplotlib.pyplot as plt

from models.gaussianprocess import GaussianProcess
from data.data import load_xy, plot_1D

if __name__ == "__main__":
    x_train, y_train = load_xy("code/data/sinusoidal_data.npz")
    plot_1D(x_train, y_train, format_str='ro')

    # length = np.exp(-1)
    # log_length = np.log(length)
    log_length = -1
    log_std = -10

    gp = GaussianProcess(
        x_train, y_train,
        log_length=log_length, log_std=log_std
    )
    x_test = np.linspace(min(x_train), max(x_train))
    y_predict = gp.predict_mean(x_test)

    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_test, y_predict, 'r--')
    plt.grid(True)
    plt.savefig("code/results/sinusoid_predictions")
    plt.close()