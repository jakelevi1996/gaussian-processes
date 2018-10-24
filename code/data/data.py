import numpy as np
import matplotlib.pyplot as plt

def load_xy(filename="code/data/cw1a.npz"):
    with np.load(filename) as data:
        x = data['x']
        y = data['y']
    
    return x, y

def convert_csv_to_npz(
    csv_filename="code/data/cw1a.csv",
    npz_filename="code/data/cw1a.npz"
):
    data = np.loadtxt(csv_filename, delimiter=',')
    n_cols = data.shape[1]
    x = data[:, :(n_cols-1)]
    y = data[:, n_cols-1]

    np.savez(npz_filename, x=x, y=y)

def plot_1D(x, y, filename='code/data/img', format_str='bx'):
    plt.figure()
    plt.plot(x, y, format_str)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()



if __name__ == "__main__":
    convert_csv_to_npz()
    x, y = load_xy()
    plot_1D(x,y)
    
    convert_csv_to_npz(
        csv_filename="code/data/cw1e.csv",
        npz_filename="code/data/cw1e.npz"
    )
    x, y = load_xy(filename="code/data/cw1e.npz")
    plot_1D(x[:,0],y, filename="code/data/2dx")
    plot_1D(x[:,1],y, filename="code/data/2dy")
