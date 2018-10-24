import numpy as np
import matplotlib.pyplot as plt

def se_iso_cov(x1, x2, log_length=-1, log_scale=0, log_std=0):
    """
    Squared exponential isotropic covariance/kernel function
    ***NOT SET UP TO COPE WITH MULTIDIMENSIONAL INPUTS.***
    np.square must be replaced with the L2 norm, and reshaping must be able to
    handle 1D arrays as well as 2D matrices
    """
    length = np.exp(log_length)
    scale = np.exp(2 * log_scale)
    noise_var = np.exp(2 * log_std)

    # x1 and x2 should be `np.array`s
    if type(x1) is not np.ndarray: x1 = np.array(x1)
    if type(x2) is not np.ndarray: x2 = np.array(x2)
    # If x1 and x2 are not matrices, assume they're column vectors
    if not len(x1.shape) == 2: x1 = x1.reshape(-1, 1)
    if not len(x2.shape) == 2: x2 = x2.reshape(-1, 1)

    # Calculate covariances:    
    k = scale * np.exp(
        -0.5 * np.square((x1 - x2.T) / length)
    ) + noise_var * np.equal(x1, x2.T)

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
        self.cov_train_inv = np.linalg.inv(self.cov_train)
    
    def kernel(self, x1, x2):
        return self.cov_func(
            x1, x2,
            log_length=self.log_length,
            log_scale=self.log_scale,
            log_std=self.log_std,
        )
    
    def update_hyperparams(
        self, log_length=None, log_scale=None, log_std=None
    ):
        # Check to see if new hyperparameters have been specified
        if any([
            log_length is not None,
            log_scale is not None,
            log_std is not None,
        ]):
            print("Updating hyperparameters")
            # have to update training covariance and inverse
            self.log_length = log_length
            self.log_scale = log_scale
            self.log_std = log_std
            # Covariance of the training inputs, using the given kernel:
            self.cov_train = self.kernel(self.x_train, self.x_train)
            self.cov_train_inv = np.linalg.inv(self.cov_train)
    
    def predict(self, x_test):
        """See equations 6.66 and 6.67 in Bishop, 2006"""
        # x_test should be an appropriate numpy.ndarray:
        if type(x_test) is not np.ndarray: x_test = np.array(x_test)
        if not len(x_test.shape) == 2: x_test = x_test.reshape(-1, 1)
        # Find the covariance between train and test inputs:
        k_predict = self.kernel(x_test, self.x_train)
        # Calculate the predictive mean:
        mean = k_predict.dot(self.cov_train_inv).dot(self.targets)

        # Find the covariance of the test inputs:
        k_test = self.kernel(x_test, x_test)
        # Calculate covariance and standard deviation:
        cov = k_test - k_predict.dot(self.cov_train_inv).dot(k_predict.T)
        # NB limited accuracy floating point arithmetic can lead to slightly
        # negative variances
        std = np.sqrt(np.maximum(np.diag(cov), 0)).reshape(mean.shape)

        return mean, std
    
    def log_evidence(self, log_length=None, log_scale=None, log_std=None):
        """See equation 6.69 of Bishop, 2006"""
        # Check to see if new hyperparameters have been specified
        self.update_hyperparams(log_length, log_scale, log_std)
        # Find the log-determinant of the covariance matrix:
        sign, logdet = np.linalg.slogdet(2 * np.pi * self.cov_train)
        # Check the determinant is positive:
        if sign <= 0: raise ValueError(
            "The covariance matrix has non-positive determinant"
        )
        # Calculate mahalanobis term for log likelihood:
        mahalanobis = self.targets.T.dot(self.cov_train_inv).dot(self.targets)
        # Return the scalar log-likelihood:
        return np.asscalar(-.5 * (logdet + mahalanobis))
    
    def grad_log_evidence(self, log_length=None, log_scale=None, log_std=None):
        """See equation 6.70 of Bishop, 2006.
        NB this method is currently only defined for the `se_iso_cov`
        covariance function."""
        # Check to see if new hyperparameters have been specified
        self.update_hyperparams(log_length, log_scale, log_std)

        # x = self.x_train
        # length = np.exp(log_length)
        # scale = np.exp(2 * log_scale)
        # noise_var = np.exp(2 * log_std)

        # mahalanobis = np.square((x - x.T) / length)




    



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
