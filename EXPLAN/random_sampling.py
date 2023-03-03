import collections
import numpy as np

def RandomSampling(instance2explain, training_data, num_samples):
    """
    This function implements random data sampling using the
    distribution of training data. instance2explain is added
    as the first row to the output.
    """

    # Collecting values and frequency of values for every feature
    feature_values = {}
    feature_frequencies = {}
    categorical_features = list(range(training_data.shape[1]))
    for feature in categorical_features:
        column = training_data[:, feature]
        feature_count = collections.Counter(column)
        values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
        feature_values[feature] = values
        feature_frequencies[feature] = (np.array(frequencies) / float(sum(frequencies)))

    # Generating random data for every feature
    random_samples = np.zeros([num_samples, instance2explain.shape[0]])
    for column in categorical_features:
        values = feature_values[column]
        frequencies = feature_frequencies[column]
        inverse_column = np.random.choice(values, size=num_samples, replace=True, p=frequencies)
        random_samples[:, column] = inverse_column
    random_samples[0] = instance2explain
    return random_samples