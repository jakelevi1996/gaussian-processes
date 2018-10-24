import numpy as np
import matplotlib.pyplot as plt

def se_iso_cov(x1, x2, log_length=-1, log_std=0):
    """ ***NOT SET UP TO COPE WITH MULTIDIMENSIONAL INPUTS. ***
    np.square must be replaced with the L2 norm, and reshaping must be able to
    handle 1D arrays as well as 2D matrices
    """
    length = np.exp(log_length)
    std = np.exp(log_std)
    
    print(x1)
    print(x2)
    if len(x1.shape) == 1:
        x1 = x1.reshape(-1, 1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(-1, 1)
    print(x1)
    print(x2)
    # x1 = x1.reshape(-1,1)
    # x2T = x2.reshape(1,-1)

    k = np.exp(-0.5 * np.square((x1 - x2.T) / length))
    k += std * np.equal(x1, x2.T)

    return k

class GaussianProcess():
    def __init__(
        self,
        x_train, y_train,
        cov_func=se_iso_cov,

    ):
        pass

if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 5, 7, 5])
    k = se_iso_cov(a, b, log_length=-20, log_std=0)
    print(k)
    
    plt.figure()
    plt.pcolor(k)
    plt.savefig("img")
    plt.close()

