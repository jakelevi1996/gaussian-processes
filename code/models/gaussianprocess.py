import numpy as np
import matplotlib.pyplot as plt

def se_iso_cov(x1, x2, log_length=-1, log_scale=0, log_std=0):
    """
    ***NOT SET UP TO COPE WITH MULTIDIMENSIONAL INPUTS.***
    np.square must be replaced with the L2 norm, and reshaping must be able to
    handle 1D arrays as well as 2D matrices
    """
    length = np.exp(log_length)
    scale = np.exp(log_scale)
    std = np.exp(log_std)

    # x1 and x2 should be `np.array`s
    if type(x1) is not np.ndarray: x1 = np.array(x1)
    if type(x2) is not np.ndarray: x2 = np.array(x2)

    # If x1 and x2 are not matrices, assume they're column vectors
    if not len(x1.shape) == 2: x1 = x1.reshape(-1, 1)
    if not len(x2.shape) == 2: x2 = x2.reshape(-1, 1)

    # Calculate covariances:    
    k = scale * np.exp(
        -0.5 * np.square((x1 - x2.T) / length)
    ) + std * np.equal(x1, x2.T)

    return k

class GaussianProcess():
    def __init__(
        self,
        x_train, y_train,
        cov_func=se_iso_cov,
        log_length=-1,
        log_scale=0,
        log_std=0,
    ):
        # x_train should be a 1D vector or 2D matrix
        if type(x_train) is not np.ndarray: x_train = np.array(x_train)
        if not len(x_train.shape) == 2: x_train = x_train.reshape(-1, 1)
        self.x_train = x_train
        
        # Targets, used for predicting means:
        self.targets = y_train.reshape(-1, 1)
        # Covariance function and hyperparameters:
        self.cov_func = cov_func
        self.log_length = log_length
        self.log_scale = log_scale
        self.log_std = log_std
        # Covariance of the training inputs, using the given kernel:
        self.cov_train = self.kernel(x_train, x_train)
        # Only invert this when it's needed:
        self.cov_train_inv = None
        # These can be used for predicting the mean for a new test-input
        # without inverting the covariance of the training points:
        self.predictive_weights = None
    
    def kernel(self, x1, x2):
        return self.cov_func(
            x1, x2,
            log_length=self.log_length,
            log_scale=self.log_scale,
            log_std=self.log_std
        )
    
    def predict(self, x_test):
        """See equations 6.66 and 6.67 in Bishop, 2006"""
        # x_test should be an appropriate numpy.ndarray:
        if type(x_test) is not np.ndarray: x_test = np.array(x_test)
        if not len(x_test.shape) == 2: x_test = x_test.reshape(-1, 1)
        # Invert the covariance matrix, if it has not been done so already:
        if not self.cov_train_inv:
            self.cov_train_inv = np.linalg.inv(self.cov_train)
        # Find the predictive weights, if they have not been found already:
        if not self.predictive_weights:
            self.predictive_weights = self.cov_train_inv.dot(self.targets)
        # Find the covariance between train and test inputs:
        k_predict = self.kernel(x_test, self.x_train)
        # Calculate the predictive mean:
        mean = k_predict.dot(self.predictive_weights)

        # Find the covariance of the test inputs:
        k_test = self.kernel(x_test, x_test)
        # Calculate covariance and standard deviation:
        cov = k_test - k_predict.dot(self.cov_train_inv).dot(k_predict.T)
        # NB Floating point arithmetic can lead to negative variances
        std = np.sqrt(np.maximum(np.diag(cov), 0)).reshape(mean.shape)

        return mean, std
    
    def predict_mean(self, x_test):
        """Just predict the mean, if the standard deviation isn't needed"""
        # x_test should be an appropriate numpy.ndarray:
        if type(x_test) is not np.ndarray: x_test = np.array(x_test)
        if not len(x_test.shape) == 2: x_test = x_test.reshape(-1, 1)

        # Find the covariance between train and test inputs:
        k_predict = self.kernel(x_test, self.x_train)
        # Fastest to use predictive weights, if they've been found:
        if self.predictive_weights:
            return k_predict.dot(self.predictive_weights)
        # Otherwise, if the training covariance has been inverted, use that:
        if self.cov_train_inv:
            return k_predict.dot(self.cov_train_inv).dot(self.targets)
        # Otherwise, find the predictive weights and predict the mean:
        self.predictive_weights = np.linalg.solve(self.cov_train, self.targets)

        return k_predict.dot(self.predictive_weights)
            



if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 5, 7, 5])
    k = se_iso_cov(a, b, log_length=0, log_std=0)
    print(k)

    k2 = se_iso_cov(2.5, a, log_length=0, log_std=0)
    print(k2)

    # k3 
    
    plt.figure()
    plt.pcolor(k)
    plt.savefig("img")
    plt.close()

