import numpy as np

def SampleManipulation(prediction, random_samples, random_samples_dc, contributions_dc):
    """
    This function uses the similarities between feature values and feature importance
    to make the neighborhood data closer to the instance2explain. The output is a
    compact data which is suitable for selecting representative data points.

    N.B.: the first row of random_samples, random_samples_dc, and contributions_dc
    correspond to instance2explain
    """

    # Creating a similarity matrix of feature contributions and filling it with 1s when
    # contributions to mutual target classes are equal, otherwise filling it with 0s.
    preds = np.argmax(prediction, axis=1)
    sim_matrix = np.zeros(random_samples.shape)
    for i in range(random_samples.shape[0]):
        for j in range(random_samples.shape[1]):
            if random_samples_dc[i, j] != random_samples_dc[0, j]:
                if preds[i] == preds[0]:
                    sim_matrix[i, j] = (contributions_dc[i, j, preds[i]] == contributions_dc[0, j, preds[i]])
                else:
                    sim_matrix[i, j] = (contributions_dc[i, j, preds[i]] == contributions_dc[0, j, preds[i]]) and \
                                       (contributions_dc[i, j, preds[0]] == contributions_dc[0, j, preds[0]])

    # Making neighborhood data closer to the instance2explain using feature importance similarity matrix
    dense_samples = random_samples.copy()
    for i in range(dense_samples.shape[0]):
        for j in range(dense_samples.shape[1]):
            dense_samples[i, j] = dense_samples[0, j] if sim_matrix[i, j] == 1 else dense_samples[i, j]
    return dense_samples
