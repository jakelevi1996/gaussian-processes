import numpy as np
import matplotlib.pyplot as plt

def load_xy(filename="code/data/cw1a.npz"):
    with np.load(filename) as data:
        x = data['x']
        y = data['y']
    
    return x, y

def plot_1D(x, y, filename='code/data/img', format_str='bx'):
    plt.figure()
    plt.plot(x, y, format_str)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def generate_sinusoid(
    filename="code/data/sinusoidal_data", n_points=50, noise_sd=0.1
):
    x = np.random.uniform(0.0, 1.0, n_points)
    y = np.sin(2*np.pi*x) + np.random.normal(scale=noise_sd, size=x.shape)
    plot_1D(x, y)
    
    np.savez(filename, x=x, y=y)

def convert_csv_to_npz(
    csv_filename="code/data/cw1a.csv",
    npz_filename="code/data/cw1a.npz"
):
    data = np.loadtxt(csv_filename, delimiter=',')
    n_cols = data.shape[1]
    x = data[:, :(n_cols-1)]
    y = data[:, n_cols-1]

    np.savez(npz_filename, x=x, y=y)



if __name__ == "__main__":
    convert_csv_to_npz()
    x, y = load_xy()
    plot_1D(x,y)
    
    generate_sinusoid()
