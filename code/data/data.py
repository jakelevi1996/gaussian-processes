import numpy as np
import matplotlib.pyplot as plt

def load_npz(filename="code/data/cw1a.npz"):
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

def save_npz(x, y, filename="code/data/data"):
    np.savez(filename, x=x, y=y)
    

def generate_sinusoid(
    filename="code/data/sinusoidal_data", n_points=50, noise_sd=0.1
):
    x = np.random.uniform(0.0, 1.0, n_points)
    y = np.sin(2*np.pi*x) + np.random.normal(scale=noise_sd, size=x.shape)
    plot_1D(x, y)
    
    save_npz(x, y, filename)

def convert_csv_to_npz(
    csv_filename="code/data/cw1a.csv",
    npz_filename="code/data/cw1a.npz"
):
    data = np.loadtxt(csv_filename, delimiter=',')
    n_cols = data.shape[1]
    x = data[:, :(n_cols-1)]
    y = data[:, n_cols-1]

    save_npz(x, y, npz_filename)

def convert_npz_to_csv(
    npz_filename="code/data/sinusoidal_data.npz",
    csv_filename="code/data/sinusoidal_data.csv"
):
    """NB ONLY WORKS FOR 1D DATA
    TODO: update for multi-D data"""
    x, y = load_npz(filename=npz_filename)
    data = np.concatenate(
        (x.reshape(-1, 1), y.reshape(-1, 1)), axis=1
    )
    np.savetxt(csv_filename, data, delimiter=', ')



if __name__ == "__main__":
    # convert_csv_to_npz()
    # x, y = load_npz()
    # plot_1D(x,y)
    
    # generate_sinusoid(n_points=80)
    
    convert_npz_to_csv()
    convert_csv_to_npz(
        "code/data/sinusoidal_data.csv",
        "code/data/sinusoidal_data2.npz"
    )
    x, y = load_npz("code/data/sinusoidal_data2.npz")
    plot_1D(x, y)
