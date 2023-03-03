import numpy as np

def QuartileDiscretization(data):
    """
    This function implements quartile discretization method.
    It is used for discretizing feature values.
    """

    bins = []
    data_dc = data.copy()
    features = list(range(data.shape[1]))
    for feature in features:
        qts = np.array(np.percentile(data[:, feature], [25, 50, 75]))
        bins.append(qts)
    bins = [np.unique(x) for x in bins]
    lambdas = {}
    for feature, qts in zip(features, bins):
        lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x)
    for feature in lambdas:
        if len(data.shape) == 1:
            data_dc[feature] = int(lambdas[feature](data_dc[feature]))
        else:
            data_dc[:, feature] = lambdas[feature](data_dc[:, feature]).astype(int)
    return data_dc
