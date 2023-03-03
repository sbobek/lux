import numpy as np
import pandas as pd

def SturgesDiscretization(data):
    """
    This function implements Sturges discretization method.
    It is used for discretizing feature importance values.
    """

    data_dc = np.zeros(data.shape)
    for l in range(data.shape[2]):  # class labels
        for j in range(data.shape[1]):  # features
            bins = np.histogram_bin_edges(data[:, j, l], bins='sturges').size
            data_dc[:, j, l] = pd.qcut(data[:, j, l], q=bins, labels=False, duplicates='drop')
    return data_dc