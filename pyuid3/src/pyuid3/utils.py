__all__ = ['StandardRescaler']


import numpy as np


class StandardRescaler:
    """Inverse transform of coefficients of a linear model.
    Parameters
    ==========
    scalar_mean : ndarray, shape=(n_features,)
        Feature-wise mean.
    scalar_scale : ndarray, shape=(n_features,)
        Feature-wise standard deviation.
    """

    def __init__(self, scalar_mean, scalar_scale):
        self.scalar_mean = scalar_mean
        self.scalar_scale = scalar_scale

    def rescale(self, coef_scalar, intercept):
        """Apply inverse transform to coefficients.
        Parameters
        ==========
        coef_scalar : ndarray, shape=(n_features,)
            Estimated coefficents after standardization.
        intercept : float
            Estimated intercept after standardization.
        Returns
        =======
        coef_rescaled : ndarray, shape=(n_features,)
            Rescaled coefficients.
        intercept_rescaled : float
            Rescaled intercept.
        """
        coef_new = coef_scalar / self.scalar_scale
        intercept_new = intercept - np.dot(coef_new, self.scalar_mean)
        return coef_new, intercept_new