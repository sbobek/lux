import time
import os
import datetime
import explan
from utils import *
from LORE import lore
from LORE.neighbor_generator import *
from lime.lime_tabular import LimeTabularExplainer
from statistics import mode
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import variation
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings("ignore")

def FidelityCoverageExperimetns(blackbox, X_explain, y_explain, index, dataset, anchor_explainer, path_data, verbose=False):
    # Reading data set information
    feature_names = dataset['feature_names']
    possible_outcomes = dataset['possible_outcomes']
    discrete_indices = dataset['discrete_indices']
    discrete_names = dataset['discrete_names']

    # Creating a data frame of the explanation data
    dfX_expalin = build_df2explain(blackbox, X_explain, dataset).to_dict('records')

    # Variable initialization
    fidelity_x_EXPLAN = exp_size_EXPLAN = cv_cv_EXPLAN = precision_EXPLAN = fidelity_X_EXPLAN = \
    coverage_EXPLAN = coverage_X_EXPLAN = n_samples_EXPLAN = distance_EXPLAN = balance_rate_X_EXPLAN = \
    fidelity_x_LORE = exp_size_LORE = cv_cv_LORE = precision_LORE = fidelity_X_LORE = coverage_LORE = \
    coverage_X_LORE = n_samples_LORE = distance_LORE = balance_rate_X_LORE = fidelity_x_Anchor = \
    exp_size_Anchor = cv_cv_Anchor =precision_Anchor = fidelity_X_Anchor = coverage_Anchor = \
    coverage_X_Anchor = n_samples_Anchor = distance_Anchor = balance_rate_X_Anchor = fidelity_x_LIME = \
    exp_size_LIME = fidelity_X_LIME = rule_LORE = rule_Anchor = rule_EXPLAN = 0

    # Hit evaluation function
    def hit_outcome(x, y):
        return 1 if x == y else 0

    # EXPLAN
    print(datetime.datetime.now(), '\tEXPLAN')
    start_time = time.time()
    try:
        # Explaining the instance specified by index
        exp_EXPLAN, info_EXPLAN = explan.Explainer(X_explain[index],
                                                   blackbox,
                                                   dataset,
                                                   N_samples=3000,
                                                   tau=250)

        # Calculating the overall neighborhood distance w.r.t instance2explain
        X = info_EXPLAN['X']
        X_hat = np.r_[X_explain[index].reshape(1, -1), X]
        distances = pairwise_distances(
            X_hat,
            X_hat[0, :].reshape(1, -1),
            metric='euclidean').ravel()
        distance_EXPLAN = np.sum(distances)

        # Calculating the feature frequency variance of neighborhood
        X_hat = X_hat[np.random.choice(range(X_hat.shape[0]),np.min([X_hat.shape[0],1000]),replace=False)]
        cv_EXPLAN = variation(X_hat, axis=0)
        cv_EXPLAN[np.isnan(cv_EXPLAN)] = 0
        cv_cv_EXPLAN = variation(cv_EXPLAN)

        # Measuring the balance rate of neighborhood samples
        n_samples_EXPLAN = X.shape[0]
        predictions = blackbox.predict(X)
        ones = np.sum(predictions)
        balance_rate_X_EXPLAN = 1 - np.abs(0.5 - (ones / n_samples_EXPLAN))

        # Extracting the predicted labels by black-box and interpretable model
        y_x_bb_EXPLAN = y_explain[index]
        y_x_dt_EXPLAN = exp_EXPLAN[0][dataset['class_name']]
        y_X_bb_EXPLAN = info_EXPLAN['y_X_bb']
        y_X_dt_EXPLAN = info_EXPLAN['y_X_dt']

        # Calculating fidelity metrics for the explained instance and its neighborhood samples
        fidelity_x_EXPLAN = hit_outcome(y_x_bb_EXPLAN, y_x_dt_EXPLAN)
        fidelity_X_EXPLAN = f1_score(y_X_bb_EXPLAN, y_X_dt_EXPLAN)

        # Printing the explanation rule
        rule_EXPLAN = exp_EXPLAN[1]
        print(rule_EXPLAN)

        # Calculating the global coverage
        covered_EXPLAN = get_covered(rule_EXPLAN, dfX_expalin, dataset)
        coverage_EXPLAN = (len(covered_EXPLAN) / len(dfX_expalin))

        # Calculating the local coverage
        covered_X_EXPLAN = get_covered(rule_EXPLAN, info_EXPLAN['dfX'].to_dict('records'), dataset)
        coverage_X_EXPLAN = (len(covered_X_EXPLAN) / len(info_EXPLAN['dfX']))

        # Measuring the precision score based on the global coverage
        precision_EXPLAN = [hit_outcome(c, y_x_dt_EXPLAN) for c in y_explain[covered_EXPLAN]]
        precision_EXPLAN = 0 if precision_EXPLAN == [] else precision_EXPLAN

        # Calculating the explanation size
        exp_size_EXPLAN = len(info_EXPLAN['tree_path']) - 1

    except Exception:
        pass

    time_EXPLAN = time.time() - start_time

    #LORE
    print(datetime.datetime.now(), '\tLORE')
    start_time = time.time()

    try:
        # Explaining the instance specified by index
        exp_LORE, info_LORE = lore.explain(index, X_explain,
                                           dataset, blackbox,
                                           ng_function=genetic_neighborhood,
                                           discrete_use_probabilities=True,
                                           continuous_function_estimation=False,
                                           returns_infos=True, path=path_data,
                                           sep=';', log=verbose)

        # Calculating the overall neighborhood distance w.r.t instance2explain
        Z = info_LORE['Z']
        Z_hat = np.r_[X_explain[index].reshape(1, -1), Z]
        distances = pairwise_distances(
            Z_hat,
            Z_hat[0, :].reshape(1, -1),
            metric='euclidean').ravel()
        distance_LORE = np.sum(distances)

        # Calculating the feature frequency variance of neighborhood
        Z_hat = Z_hat[np.random.choice(range(Z_hat.shape[0]), np.min([Z_hat.shape[0], 1000]), replace=False)]
        cv_LORE = variation(Z_hat, axis=0)
        cv_LORE[np.isnan(cv_LORE)] = 0
        cv_cv_LORE = variation(np.abs(cv_LORE))

        # Measuring the balance rate of neighborhood samples
        n_samples_LORE = Z.shape[0]
        predictions = blackbox.predict(Z)
        ones = np.sum(predictions)
        balance_rate_X_LORE = 1 - np.abs(0.5 - (ones / n_samples_LORE))

        # Extracting the predicted labels by black-box and interpretable model
        y_x_bb_LORE = y_explain[index]
        y_x_dt_LORE = exp_LORE[0][0][dataset['class_name']]
        y_X_bb_LORE = info_LORE['y_pred_bb']
        y_X_dt_LORE = info_LORE['y_pred_cc']

        # Calculating fidelity metrics for the explained instance and its neighborhood samples
        fidelity_x_LORE = hit_outcome(y_x_bb_LORE, y_x_dt_LORE)
        fidelity_X_LORE = f1_score(y_X_bb_LORE, y_X_dt_LORE)

        # Printing the explanation rule
        rule_LORE = exp_LORE[0][1]
        print(rule_LORE)

        # Calculating the global coverage
        covered_LORE = get_covered(rule_LORE, dfX_expalin, dataset)
        coverage_LORE  = (len(covered_LORE ) / len(dfX_expalin))

        # Calculating the local coverage
        covered_X_LORE  = get_covered(rule_LORE, info_LORE['dfZ'].to_dict('records'), dataset)
        coverage_X_LORE  = (len(covered_X_LORE ) / len(info_LORE['dfZ']))

        # Measuring the precision score based on the global coverage
        precision_LORE  = [hit_outcome(c, y_x_dt_LORE) for c in y_explain[covered_LORE]]
        precision_LORE  = 0 if precision_LORE  == [] else precision_LORE

        # Calculating the explanation size
        exp_size_LORE = len(info_LORE['tree_path']) - 1

    except Exception:
        pass

    time_LORE = time.time() - start_time

    # Anchor
    print(datetime.datetime.now(), '\tAnchor')
    start_time = time.time()
    try:
        # Explaining the instance specified by index
        exp_Anchor, info_Anchor = anchor_explainer.explain_instance(X_explain[index].reshape(1, -1),
                                                                    blackbox.predict, threshold=0.95)

        # Calculating the overall neighborhood distance w.r.t instance2explain
        Z = info_Anchor['state']['raw_data']
        Z = Z[:info_Anchor['state']['current_idx'] - 1, :]
        Z_hat = np.r_[X_explain[index].reshape(1, -1), Z]
        distances = pairwise_distances(
            Z_hat,
            Z_hat[0, :].reshape(1, -1),
            metric='euclidean').ravel()
        distance_Anchor = np.sum(distances)

        # Calculating the feature frequency variance of neighborhood
        Z_hat = Z_hat[np.random.choice(range(Z_hat.shape[0]), np.min([Z_hat.shape[0], 1000]), replace=False)]
        cv_Anchor = variation(Z_hat, axis=0)
        cv_Anchor[np.isnan(cv_Anchor)] = 0
        cv_cv_Anchor = variation(cv_Anchor)

        # Measuring the balance rate of neighborhood samples
        n_samples_Anchor = Z.shape[0]
        predictions = blackbox.predict(Z)
        ones = np.sum(predictions)
        balance_rate_X_Anchor = 1 - np.abs(0.5 - (ones / n_samples_Anchor))

        # Extracting the predicted labels by black-box and interpretable model
        y_X_bb_Anchor = blackbox.predict(Z)
        y_X_dt_Anchor = blackbox.predict(Z)
        y_x_bb_Anchor = y_explain[index]

        # Printing the explanation rule
        rule_Anchor = anchor2arule(exp_Anchor)
        print(rule_Anchor)

        # Calculating the global coverage
        covered_Anchor = get_covered(rule_Anchor, dfX_expalin, dataset)
        coverage_Anchor = (len(covered_Anchor) / len(dfX_expalin))

        # Calculating fidelity metrics for the explained instance and its neighborhood samples
        if len(covered_Anchor) > 0:
            if isinstance(y_explain[0], str):
                y_x_dt_Anchor = mode(y_explain[covered_Anchor])
            else:
                y_x_dt_Anchor = int(np.round(y_explain[covered_Anchor].mean()))
        else:
            y_x_dt_Anchor = y_x_bb_Anchor
        fidelity_x_Anchor = hit_outcome(y_x_bb_Anchor, y_x_dt_Anchor)
        fidelity_X_Anchor = f1_score(y_X_bb_Anchor, y_X_dt_Anchor)

        # Calculating the local coverage
        dfZ = build_df2explain(blackbox, Z, dataset).to_dict('records')
        covered_X_Anchor = get_covered(rule_Anchor, dfZ, dataset)
        coverage_X_Anchor = (len(covered_X_Anchor) / len(Z))

        # Measuring the precision score based on the global coverage
        precision_Anchor = [hit_outcome(v, y_x_dt_Anchor) for v in y_explain[covered_Anchor]]
        precision_Anchor = 0 if precision_Anchor == [] else precision_Anchor

        # Calculating the explanation size
        exp_size_Anchor = len(rule_Anchor)

    except Exception:
        pass

    time_Anchor = time.time() - start_time

    # LIME
    print(datetime.datetime.now(), '\tLIME')
    start_time = time.time()
    try:
        # Creating LIME tabular explainer
        exp_LIME =  LimeTabularExplainer(X_explain,
                                        feature_names=feature_names,
                                        class_names=possible_outcomes,
                                        categorical_features=discrete_indices,
                                        categorical_names=discrete_names,
                                        verbose=False)
        # Finding the number of explanation features that result
        # in the highest score of the interpretable mode
        score = []
        for i in range(2, 11):
            exp, Zlr, Z, lr = exp_LIME.explain_instance(X_explain[index],
                                                        blackbox.predict_proba,
                                                        num_features=i,
                                                        num_samples=5000)
            score.append(exp.score)
        num_features = score.index(max(score)) + 2

        # Explaining the instance using the best number of features
        exp, Zlr, Z, lr = exp_LIME.explain_instance(X_explain[index],
                                                    blackbox.predict_proba,
                                                    num_features=num_features,
                                                    num_samples=5000)

        # Extracting the information provided by the feature importance explanation
        used_features_idx = list()
        used_features_importance = list()
        logic_explanation = list()
        for idx, weight in exp.local_exp[1]:
            used_features_idx.append(idx)
            used_features_importance.append(weight)
            logic_explanation.append(exp.domain_mapper.discretized_feature_names[idx])

        # Printing the feature importance explanation
        for feature, weight in zip(logic_explanation, used_features_importance):
            print(feature, weight)

        # Extracting the predicted labels by black-box and interpretable model
        y_x_bb_LIME = blackbox.predict(Z[0].reshape(1, -1))[0]
        y_x_lr_LIME = np.round(lr.predict(Zlr[0, used_features_idx].reshape(1, -1))).astype(int)[0]
        y_X_bb_LIME = blackbox.predict(Z)
        y_X_lr_LIME = np.round(lr.predict(Zlr[:, used_features_idx])).astype(int)

        # Calculating fidelity metrics for the explained instance and its neighborhood samples
        fidelity_x_LIME = hit_outcome(y_x_bb_LIME, y_x_lr_LIME)
        fidelity_X_LIME = f1_score(y_X_bb_LIME, y_X_lr_LIME)

        # Calculating the explanation size
        exp_size_LIME = num_features

    except Exception:
        pass

    time_LIME = time.time() - start_time

    # Returning the achieved results
    results = '%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%.3f,%.3f,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%.3f,%.3f,' \
          '%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%.3f,%.3f,%d,%d,%.3f,%.3f,%s,%s,%s,%s,%s,%s' \
          % (fidelity_x_EXPLAN, exp_size_EXPLAN, cv_cv_EXPLAN, np.mean(precision_EXPLAN), fidelity_X_EXPLAN,
            coverage_EXPLAN, coverage_X_EXPLAN, n_samples_EXPLAN, distance_EXPLAN, balance_rate_X_EXPLAN, time_EXPLAN,
            fidelity_x_LORE, exp_size_LORE, cv_cv_LORE, np.mean(precision_LORE), fidelity_X_LORE,
            coverage_LORE, coverage_X_LORE, n_samples_LORE, distance_LORE, balance_rate_X_LORE, time_LORE,
            fidelity_x_Anchor, exp_size_Anchor, cv_cv_Anchor, np.mean(precision_Anchor), fidelity_X_Anchor,
            coverage_Anchor, coverage_X_Anchor, n_samples_Anchor, distance_Anchor, balance_rate_X_Anchor, time_Anchor,
            fidelity_x_LIME, exp_size_LIME, fidelity_X_LIME, time_LIME,
            'EXPLAN Rule -> ', rule_EXPLAN, 'LORE Rule ->', rule_LORE, 'Anchor Rule ->', rule_Anchor)
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
        'adult': ('adult.csv', prepare_adult_dataset),
    }

    # Defining the list of black-boxes
    blackbox_list = {
        'lr': LogisticRegression,
        'gt': GradientBoostingClassifier,
        'nn': MLPClassifier,
    }

    for dataset_kw in datsets_list:
        # Reading a data set
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_name, path_data)

        # Splitting the data set into train, test, and explain sets
        X, y = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_explain = X_test[:int(len(X_test) / 2), :]

        # Creating Anchor explainer
        anchor_explainer = fit_anchor(dataset, X_train, X_test, y_train, y_test, X_explain)

        for blackbox_name in blackbox_list:
            # Creating a csv file corresponding to the data set and black-box
            exists = os.path.isfile(path_exp + 'fidelity_coverage_%s_%s.csv' % (dataset_kw, blackbox_name))
            if exists:
                os.remove(path_exp + 'fidelity_coverage_%s_%s.csv' % (dataset_kw, blackbox_name))
            experiment_results = open(path_exp + 'fidelity_coverage_%s_%s.csv' % (dataset_kw, blackbox_name), 'a')


            # Adding headers to the csv file
            headers = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                  '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('', '', '',
                    'fidelity_x_EXPLAN', 'exp_size_EXPLAN', 'cv_cv_EXPLAN', 'precision_EXPLAN', 'fidelity_X_EXPLAN',
                    'coverage_EXPLAN', 'coverage_X_EXPLAN', 'n_samples_EXPLAN', 'distance_EXPLAN', 'balance_rate_X_EXPLAN',
                    'time_EXPLAN', 'fidelity_x_LORE', 'exp_size_LORE', 'cv_cv_LORE', 'precision_LORE', 'fidelity_X_LORE',
                    'coverage_LORE', 'coverage_X_LORE', 'n_samples_LORE', 'distance_LORE', 'balance_rate_X_LORE', 'time_LORE',
                    'fidelity_x_Anchor', 'exp_size_Anchor', 'cv_cv_Anchor', 'precision_Anchor', 'fidelity_X_Anchor',
                    'coverage_Anchor', 'coverage_X_Anchor', 'n_samples_Anchor', 'distance_Anchor', 'balance_rate_X_Anchor',
                    'time_Anchor', 'fidelity_x_LIME', 'exp_size_LIME', 'fidelity_X_LIME', 'time_LIME')
            experiment_results.write(headers)


            # Adding average functions to the csv file
            average = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                  '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('', '', 'AVERAGE',
                 '=average(D4:D4000)', '=average(E4:E4000)', '=average(F4:F4000)', '=average(G4:G4000)',
                 '=average(H4:H4000)', '=average(I4:I4000)', '=average(J4:J4000)', '=average(K4:K4000)',
                 '=average(L4:L4000)', '=average(M4:M4000)', '=average(N4:N4000)', '=average(O4:O4000)',
                 '=average(P4:P4000)', '=average(Q4:Q4000)', '=average(R4:R4000)', '=average(S4:S4000)',
                 '=average(T4:T4000)', '=average(U4:U4000)', '=average(V4:V4000)', '=average(W4:W4000)',
                 '=average(X4:X4000)', '=average(Y4:Y4000)', '=average(Z4:Z4000)', '=average(AA4:AA4000)',
                 '=average(AB4:AB4000)', '=average(AC4:AC4000)', '=average(AD4:AD4000)', '=average(AE4:AE4000)',
                 '=average(AF4:AF4000)', '=average(AG4:AG4000)', '=average(AH4:AH4000)', '=average(AI4:AI4000)',
                 '=average(AJ4:AJ4000)', '=average(AK4:AK4000)', '=average(AL4:AL4000)', '=average(AM4:AM4000)',
                 '=average(AN4:AN4000)')
            experiment_results.write(average)

            # Adding standard deviation functions to the csv file
            stddev = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,' \
                  '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('', '', 'STDDEV',
                  '=stdev(D4:D4000)', '=stdev(E4:E4000)', '=stdev(F4:F4000)', '=stdev(G4:G4000)',
                  '=stdev(H4:H4000)', '=stdev(I4:I4000)', '=stdev(J4:J4000)', '=stdev(K4:K4000)',
                  '=stdev(L4:L4000)', '=stdev(M4:M4000)', '=stdev(N4:N4000)', '=stdev(O4:O4000)',
                  '=stdev(P4:P4000)', '=stdev(Q4:Q4000)', '=stdev(R4:R4000)', '=stdev(S4:S4000)',
                  '=stdev(T4:T4000)', '=stdev(U4:U4000)', '=stdev(V4:V4000)', '=stdev(W4:W4000)',
                  '=stdev(X4:X4000)', '=stdev(Y4:Y4000)', '=stdev(Z4:Z4000)', '=stdev(AA4:AA4000)',
                  '=stdev(AB4:AB4000)', '=stdev(AC4:AC4000)', '=stdev(AD4:AD4000)', '=stdev(AE4:AE4000)',
                  '=stdev(AF4:AF4000)', '=stdev(AG4:AG4000)', '=stdev(AH4:AH4000)', '=stdev(AI4:AI4000)',
                  '=stdev(AJ4:AJ4000)', '=stdev(AK4:AK4000)', '=stdev(AL4:AL4000)', '=stdev(AM4:AM4000)',
                  '=stdev(AN4:AN4000)')
            experiment_results.write(stddev)

            # Creating and training black-box
            BlackBoxConstructor = blackbox_list[blackbox_name]
            blackbox = BlackBoxConstructor(random_state=42)
            blackbox.fit(X_train, y_train)

            # Achieving black-box labels of the instances in the explain set
            y_explain = blackbox.predict(X_explain)
            y_explain = np.asarray([dataset['label_encoder'][dataset['class_name']].classes_[i] for i in y_explain])

            # Running fidelity and coverage experiment for every instance in X_explain
            for index in range(len(X_explain)):
                if index <= start_index:
                    continue
                print(datetime.datetime.now(), '%d - %.2f' % (index, index / len(X_explain)))
                results = FidelityCoverageExperimetns(blackbox, X_explain, y_explain, index, dataset,
                                                      anchor_explainer, path_data, verbose=False)

                # Adding results to the csv file
                results = '%d,%s,%s,%s\n' % (index, dataset_kw, blackbox_name, results)
                experiment_results.write(results)
                experiment_results.flush()
            experiment_results.close()

if __name__ == "__main__":
    main()
