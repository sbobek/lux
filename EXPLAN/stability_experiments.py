import os
import datetime
import explan
from utils import *
from LORE import lore
from LORE.neighbor_generator import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

def StabilityExperiment(blackbox, X_explain, index, dataset, anchor_explainer, path_data, verbose=False):
    # Number of runs
    n_run = 5

    # Variable initialization
    jaccard_features_LORE = list()
    similar_feature_values_LORE = list()
    deviation_n_features_LORE = list()

    jaccard_features_Anchor = list()
    similar_feature_values_Anchor = list()
    deviation_n_features_Anchor = list()

    jaccard_features_EXPLAN = list()
    similar_feature_values_EXPLAN = list()
    deviation_n_features_EXPLAN = list()

    # EXPLAN
    print(datetime.datetime.now(), '\tEXPLAN')
    feature_names_EXPLAN = list()
    feature_values_EXPLAN = list()
    n_features_EXPLAN = list()
    for n in range(n_run):
        # Explaining the instance specified by index
        exp_EXPLAN, info_EXPLAN = explan.Explainer(X_explain[index],
                                                   blackbox,
                                                   dataset,
                                                   N_samples=3000,
                                                   tau=250)
        # Extracting feature names, feature values, and
        # number of features from the explanation rule
        rule_EXPLAN = exp_EXPLAN[1]
        feature_names_EXPLAN.append(list(rule_EXPLAN.keys()))
        feature_values_EXPLAN.append(rule_EXPLAN)
        n_features_EXPLAN.append(len(list(rule_EXPLAN.keys())))
    print(feature_names_EXPLAN)

    # LORE
    print(datetime.datetime.now(), '\tLORE')
    feature_names_LORE = list()
    feature_values_LORE = list()
    n_features_LORE = list()
    for n in range(n_run):
        while True:
            try:
                # Explaining the instance specified by index
                exp_LORE, info_LORE = lore.explain(index, X_explain,
                                                   dataset, blackbox,
                                                   ng_function=genetic_neighborhood,
                                                   discrete_use_probabilities=True,
                                                   continuous_function_estimation=False,
                                                   returns_infos=True, path=path_data,
                                                   sep=';', log=verbose)
                if exp_LORE[1] != []:
                    break

            except Exception:
                pass
        # Extracting feature names, feature values, and
        # number of features from the explanation rule
        rule_LORE = exp_LORE[0][1]
        feature_names_LORE.append(list(rule_LORE.keys()))
        feature_values_LORE.append(rule_LORE)
        n_features_LORE.append(len(list(rule_LORE.keys())))
    print(feature_names_LORE)

    # Anchor
    print(datetime.datetime.now(), '\tAnchor')
    feature_names_Anchor = list()
    feature_values_Anchor = list()
    n_features_Anchor = list()
    for n in range(n_run):
        # Explaining the instance specified by index
        exp_Anchor, info_Anchor = anchor_explainer.explain_instance(X_explain[index].reshape(1, -1),
                                                                    blackbox.predict,
                                                                    threshold=0.95)
        # Extracting feature names, feature values, and
        # number of features from the explanation rule
        rule_Anchor = anchor2arule(exp_Anchor)
        feature_names_Anchor.append(list(rule_Anchor.keys()))
        feature_values_Anchor.append(rule_Anchor)
        n_features_Anchor.append(len(list(rule_Anchor.keys())))
    print(feature_names_Anchor)

    # Calculating explanation comparison metrics
    for i in range(0, 10):
        for ii in range(i, 10):

            if len(feature_names_EXPLAN) > ii:
                # Calculating Jaccard similarity between feature names of the predicates of the rules
                jaccard = len(set(feature_names_EXPLAN[i]) & set(feature_names_EXPLAN[ii])) / \
                     len(set(feature_names_EXPLAN[i]) | set(feature_names_EXPLAN[ii]))
                jaccard_features_EXPLAN.append(jaccard)

                # Calculating the similarity between feature values of predicates of the rules
                similarity = [1 if feature_values_EXPLAN[i][f] == feature_values_EXPLAN[ii][f] else 0
                      for f in set(feature_names_EXPLAN[i]) & set(feature_names_EXPLAN[ii])]
                [similar_feature_values_EXPLAN.append(s) for s in similarity]

                # Calculating the deviation from the number of predicates in the collected rules
                deviation = np.abs(n_features_EXPLAN[i] - n_features_EXPLAN[ii])
                deviation_n_features_EXPLAN.append(deviation)

            if len(feature_names_LORE) > ii:
                # Calculating Jaccard similarity between feature names of the predicates of the rules
                jaccard = len(set(feature_names_LORE[i]) & set(feature_names_LORE[ii])) / \
                     len(set(feature_names_LORE[i]) | set(feature_names_LORE[ii]))
                jaccard_features_LORE.append(jaccard)

                # Calculating the similarity between feature values of predicates of the rules
                similarity = [1 if feature_values_LORE[i][f] == feature_values_LORE[ii][f] else 0
                      for f in set(feature_names_LORE[i]) & set(feature_names_LORE[ii])]
                [similar_feature_values_LORE.append(s) for s in similarity]

                # Calculating the deviation from the number of predicates in the collected rules
                deviation = np.abs(n_features_LORE[i] - n_features_LORE[ii])
                deviation_n_features_LORE.append(deviation)

            if len(feature_names_Anchor) > ii:
                # Calculating Jaccard similarity between feature names of the predicates of the rules
                jaccard = len(set(feature_names_Anchor[i]) & set(feature_names_Anchor[ii])) / \
                     len(set(feature_names_Anchor[i]) | set(feature_names_Anchor[ii]))
                jaccard_features_Anchor.append(jaccard)

                # Calculating the similarity between feature values of predicates of the rules
                similarity = [1 if feature_values_Anchor[i][f] == feature_values_Anchor[ii][f] else 0
                      for f in set(feature_names_Anchor[i]) & set(feature_names_Anchor[ii])]
                [similar_feature_values_Anchor.append(s) for s in similarity]

                # Calculating the deviation from the number of predicates in the collected rules
                deviation = np.abs(n_features_Anchor[i] - n_features_Anchor[ii])
                deviation_n_features_Anchor.append(deviation)

    # Returning the achieved results
    results = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (
        np.mean(jaccard_features_EXPLAN),
        np.mean(similar_feature_values_EXPLAN),
        np.mean(deviation_n_features_EXPLAN),
        np.mean(jaccard_features_LORE),
        np.mean(similar_feature_values_LORE),
        np.mean(deviation_n_features_LORE),
        np.mean(jaccard_features_Anchor),
        np.mean(similar_feature_values_Anchor),
        np.mean(deviation_n_features_Anchor))
    return results

def main():

    # Defining path of data sets and experiment results
    start_index = -1
    path = './'
    path_data = path + 'datasets/'
    path_exp = path + 'experiments/'

    # Defining the list of data sets
    datsets_list = {
        'german': ('german_credit.csv', prepare_german_dataset),
        'compas': ('compas-scores-two-years.csv', prepare_compass_dataset),
        'adult': ('adult.csv', prepare_adult_dataset)
    }

    # Defining the list of black-boxes
    blackbox_list = {
        'lr': LogisticRegression,
        'gt': GradientBoostingClassifier,
        'nn': MLPClassifier
    }

    for dataset_kw in datsets_list:
        # Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_name, path_data)

        # Splitting the data set into train, test, and explain sets
        X, y = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_explain = X_test[:int(len(X_test)/2),:]

        # Creating Anchor explainer
        anchor_explainer = fit_anchor(dataset, X_train, X_test, y_train, y_test, X_explain)

        for blackbox_name in blackbox_list:
            # Creating a csv file corresponding to the data set and black-box
            exists = os.path.isfile(path_exp + 'stability_%s_%s.csv' % (dataset_kw, blackbox_name))
            if exists:
                os.remove(path_exp + 'stability_%s_%s.csv' % (dataset_kw, blackbox_name))
            experiment_results = open(path_exp + 'stability_%s_%s.csv' % (dataset_kw, blackbox_name), 'a')

            # Adding headers to the csv file
            headers = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('', '', '',
                      'jaccard_features_EXPLAN', 'similar_feature_values_EXPLAN', 'deviation_n_features_EXPLAN',
                      'jaccard_features_LORE', 'similar_feature_values_LORE', 'deviation_n_features_LORE',
                      'jaccard_features_Anchor', 'similar_feature_values_Anchor', 'deviation_n_features_Anchor')
            experiment_results.write(headers)

            # Adding average functions to the csv file
            average = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('', '', 'AVERAGE',
                 '=average(D4:D4000)', '=average(E4:E4000)', '=average(F4:F4000)',
                 '=average(G4:G4000)', '=average(H4:H4000)', '=average(I4:I4000)',
                 '=average(J4:J4000)', '=average(K4:K4000)', '=average(L4:L4000)')
            experiment_results.write(average)

            # Adding standard deviation functions to the csv file
            stddev = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('', '', 'STDDEV',
                  '=stdev(D4:D4000)', '=stdev(E4:E4000)', '=stdev(F4:F4000)',
                  '=stdev(G4:G4000)', '=stdev(H4:H4000)', '=stdev(I4:I4000)',
                  '=stdev(J4:J4000)', '=stdev(K4:K4000)', '=stdev(L4:L4000)')
            experiment_results.write(stddev)

            # Creating and training black-box
            BlackBoxConstructor = blackbox_list[blackbox_name]
            blackbox = BlackBoxConstructor(random_state=42)
            blackbox.fit(X_train, y_train)

            # Running stability experiment for every instance in X_explain
            for index in range(len(X_explain)):
                if index <= start_index:
                    continue
                print(datetime.datetime.now(), '%d - %.2f' % (index, index / len(X_explain)))
                results = StabilityExperiment(blackbox, X_explain, index, dataset, anchor_explainer,
                                              path_data, verbose=False)

                # Adding results to the csv file
                results = '%d,%s,%s,%s\n' % (index, dataset_kw, blackbox_name, results)
                experiment_results.write(results)
                experiment_results.flush()
            experiment_results.close()

if __name__ == "__main__":
    main()
