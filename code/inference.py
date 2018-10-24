import numpy as np
import matplotlib.pyplot as plt

from models.gaussianprocess import GaussianProcess
from data.data import load_xy, plot_1D

def plot_predictions(
    x_train, y_train, x_test,
    mean, std, std_scale=1.96,
    filename="code/results/sinusoid_predictions"
):
    plt.figure()
    plt.plot(x_train, y_train, 'rx')
    plt.plot(x_test, mean, 'b-')
    plt.fill_between(
        x_test,
        (mean + std_scale*std).reshape(-1),
        (mean - std_scale*std).reshape(-1),
        alpha=0.2
    )
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def gaussian_density(x, mean, std):
    return np.exp(
        -0.5 * np.square((x - mean) / std)
    ) / np.sqrt(2 * np.pi * std)

def plot_density(
    x_train, y_train, x_test,
    mean, std,
    filename="code/results/sinusoid_predictions"
):
    pass

if __name__ == "__main__":
    x_train, y_train = load_xy("code/data/sinusoidal_data.npz")
    plot_1D(x_train, y_train, format_str='ro')

    log_length = -1
    log_scale = 2
    log_std = -5

    gp = GaussianProcess(
        x_train, y_train,
        log_length=log_length, log_scale=log_scale, log_std=log_std
    )

    x_test = np.linspace(0, 1, 200)
    mean, std = gp.predict(x_test)

    plot_predictions(x_train, y_train, x_test, mean, std)
    