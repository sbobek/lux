import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def stability(rules_and_instances,dataset, features, categorical_indicator):
    """
    Calculate stability metrics for a set of rules and instances.

    :param rules_and_instances: pandas.DataFrame
        A DataFrame containing rules and instances. It should include at least the following columns:
        - 'true_class': The true class label for each instance.
        - 'explain_instance': The explanation instance corresponding to each rule.
    :type rules_and_instances: pandas.DataFrame
    :param dataset: pandas.DataFrame
        The dataset on which the rules are applied. It should be a pandas DataFrame.
    :type dataset: pandas.DataFrame
    :param features: list
        A list of feature names in the dataset.
    :type features: list
    :param categorical_indicator: list
        A list indicating whether each feature is categorical or not. Each element of the list
        corresponds to a feature in `features`, with `True` indicating the feature is categorical
        and `False` indicating it is not.
    :type categorical_indicator: list

    :return: tuple of floats
        A tuple containing the following stability metrics:
        - Mean stability of rules for each true class label.
        - Mean standard deviation of stability for each true class label.
        - Mean feature Jaccard similarity coefficient between rules.
        - Standard deviation of feature Jaccard similarity coefficient between rules.
    :rtype: tuple

    :notes:
        - Stability is calculated based on the Jaccard similarity coefficient between rules applied
          to similar instances.

        - Feature Jaccard similarity coefficient measures the similarity of features between rules.

        - Large stability and low variance are desired for stable explanations.
    """
    results_mean =[]
    results_std =[]
    feature_jackart = np.zeros((rules_and_instances.shape[0],rules_and_instances.shape[0]))
    for true_class in rules_and_instances['true_class'].unique():
        rules = rules_and_instances[rules_and_instances['true_class']==true_class]
        instance_similarity = sklearn.metrics.pairwise_distances(np.concatenate(rules['explain_instance'].values))
        jackart = np.zeros((rules.shape[0],rules.shape[0]))
        for i in range(rules.shape[0]):
            for ii in range(rules.shape[0]):
                jackart[i,ii] = average_jackart(rules.iloc[i,:], rules.iloc[ii,:], dataset, features, categorical_indicator)
        stab = jackart/(instance_similarity+1)
        results_mean.append(np.mean(stab))
        results_std.append(np.std(stab))
    for i in range(rules_and_instances.shape[0]):
        for ii in range(rules_and_instances.shape[0]):
            rule_1 = rules_and_instances.iloc[i,:]
            rule_2 = rules_and_instances.iloc[ii,:]
            if len(set(rule_1.keys())|set(rule_2.keys())) == 0:
                feature_jackart[i,ii] = 0
            else:
                feature_jackart[i,ii] = len(set(rule_1.keys())&set(rule_2.keys()))/len(set(rule_1.keys())|set(rule_2.keys()))

    return (np.mean(results_mean), np.mean(results_std),np.mean(feature_jackart),np.std(feature_jackart)) #large stability, low variance is desired


def local_fidelity(rule, dataset, features, categorical_indicator, prediction,
                       class_label='class', average='micro'):
    """
    Calculate coverage and various evaluation metrics for a given rule applied to a dataset.

    :param rule: A dictionary representing the rule to be evaluated.
                 Each key corresponds to a feature, and the corresponding value is a list of conditions
                 applied to that feature.
    :type rule: dict
    :param dataset: The dataset on which the rule is applied. It should be a pandas DataFrame.
    :type dataset: pandas.DataFrame
    :param features: A list of feature names in the dataset.
    :type features: list
    :param categorical_indicator: A list indicating whether each feature is categorical or not.
                                   Each element of the list corresponds to a feature in `features`, with `True`
                                   indicating the feature is categorical and `False` indicating it is not.
    :type categorical_indicator: list
    :param prediction: The prediction value assigned to instances covered by the rule.
    :type prediction: int or float
    :param class_label: The name of the column containing the class labels in the dataset.
                         Default is 'class'.
    :type class_label: str, optional
    :param average: The averaging strategy for multiclass classification metrics.
                     It can be one of {'micro', 'macro', 'weighted'}. Default is 'micro'.
    :type average: str, optional

    :return: A tuple containing:
             - coverage_ratio: The ratio of instances covered by the rule to the total instances in the dataset.
             - accuracy: The accuracy of the rule on the covered instances.
             - precision: The precision of the rule on the covered instances.
             - recall: The recall of the rule on the covered instances.
             - f1_score: The F1-score of the rule on the covered instances.
    :rtype: tuple

    :notes:

    - If `rule` contains conditions for features that are not present in the dataset, those conditions will be ignored.
    """
    query = []
    if rule == {}:
        return 0, 0
    for i, v in rule.items():
        op = '' if dict(zip(features, categorical_indicator))[i] == False else '=='
        query.append(f'{i}{op}' + f' and {i}{op}'.join(v))
    print(' and '.join(query))
    covered = dataset.query(' and '.join(query))
    predictions = np.ones(covered[class_label].shape[0]) * float(prediction)

    accuracy = accuracy_score(covered[class_label], predictions)
    precision = precision_score(covered[class_label], predictions, average=average)
    recall = recall_score(covered[class_label], predictions, average=average)
    f1 = f1_score(covered[class_label], predictions, average=average)

    return len(covered) / len(dataset), accuracy, precision, recall, f1


def average_jackart(rule_1, rule_2, dataset, features, categorical_indicator):
    """
        Calculate the average Jaccard similarity coefficient between two sets of rules.

        :param rule_1: A dictionary representing the first set of rules.
                       Each key corresponds to a feature, and the corresponding value is a list of conditions
                       applied to that feature.
        :type rule_1: dict
        :param rule_2:
           A dictionary representing the second set of rules. Similar to `rule_1`.
        :type rule_2: dict
        :param dataset:
           The dataset on which the rules are applied. It should be a pandas DataFrame.
        :type dataset: pandas.DataFrame
        :param features:
           A list of feature names in the dataset.
        :type features: list
        :param categorical_indicator:
           A list indicating whether each feature is categorical or not.
           Each element of the list corresponds to a feature in `features`, with `True`
           indicating the feature is categorical and `False` indicating it is not.
        :type categorical_indicator: list
        :return:
            The average Jaccard similarity coefficient between the rules in `rule_1` and `rule_2`.
            If there are no rules in either `rule_1` or `rule_2`, the function returns 0.
        :rtype: float
        :notes:

        - If `rule_1` or `rule_2` contains rules for features that are not present in the dataset, those rules will be ignored.

        - The function handles cases where either `rule_1` or `rule_2` is empty by returning 0.

        - The Jaccard similarity coefficient is calculated between the values of the features specified in the rules.
          If both `rule_1` and `rule_2` contain rules for the same feature, the coefficient is calculated between
          the corresponding values.
        """
    total_jackart = 0
    for i, v in rule_1.items():
        op = '' if dict(zip(features, categorical_indicator))[i] == False else '=='
        v1 = dataset.query(f'{i}{op}' + f'and {i}{op}'.join(v))[i]
        if i in rule_2.keys():
            v2 = dataset.query(f'{i}{op}' + f'and {i}{op}'.join(rule_2[i]))[i]
            if len((set(v1) | set(v2))) == 0:
                jackard = 0
            else:
                jackard = len(set(v1) & set(v2)) / len((set(v1) | set(v2)))
        else:
            jackard = 0
        total_jackart += jackard
    div = len(set(rule_1.keys()) | set(rule_2.keys()))
    if div == 0:
        return 0
    else:
        return total_jackart / div

