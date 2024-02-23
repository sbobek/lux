from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import pandas as pd

from lux.UncertainSMOTE import UncertainSMOTE
from lux.pyuid3.data import Data
from lux.pyuid3.entropy_evaluator import UncertainEntropyEvaluator
from lux.pyuid3.uid3 import UId3
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS
import shap
import sklearn
import gower
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
import numdifftools as nd

"""
This module contains functions that implement generation of local rule-based model-agnostic explanations.

"""


class LUX(BaseEstimator):
    REPRESENTATIVE_CENTROID = "centroid"
    REPRESENTATIVE_NEAREST = "nearest"

    CF_REPRESENTATIVE_MEDOID = "medoid"
    CF_REPRESENTATIVE_NEAREST = "nearest"

    OS_STRATEGY_SMOTE = 'smote'
    OS_STRATEGY_IMPORTANCE = 'importance'
    OS_STRATEGY_BOTH = 'both'

    def __init__(self, predict_proba, classifier=None, neighborhood_size=0.1, max_depth=2, node_size_limit=1,
                 grow_confidence_threshold=0, min_impurity_decrease=0, min_samples=5, min_generate_samples=0.02,
                 uncertainty_sigma=2, oversampling_strategy='smote'):
        """

        :param predict_proba:
        :param classifier:
        :param neighborhood_size:
        :param max_depth:
        :param node_size_limit:
        :param grow_confidence_threshold:
        :param min_impurity_decrease:
        :param min_samples:
        :param min_generate_samples:
        :param uncertainty_sigma:
        :param oversampling_strategy:
        """
        self.neighborhood_size = neighborhood_size
        self.max_depth = max_depth
        self.node_size_limit = node_size_limit
        self.grow_confidence_threshold = grow_confidence_threshold
        self.predict_proba = predict_proba
        self.attributes_names = None
        self.min_impurity_decrease = min_impurity_decrease
        self.classifier = classifier
        self.min_samples = min_samples
        self.categorical = None
        self.min_generate_samples = min_generate_samples
        self.oversampling_strategy = oversampling_strategy
        self.uncertainty_sigma = uncertainty_sigma

        if classifier is None:
            self.oversampling_strategy = self.OS_STRATEGY_SMOTE

    def fit(self, X, y, instance_to_explain, X_importances=None, exclude_neighbourhood=False, use_parity=True,
            parity_strategy='global', inverse_sampling=False, class_names=None, discount_importance=False,
            uncertain_entropy_evaluator=UncertainEntropyEvaluator(), beta=1, representative='centroid',
            density_sampling=False, radius_sampling=False, oversampling=False, categorical=None, prune=False,
            oblique=False, n_jobs=None):
        """

        :param X:
        :param y:
        :param instance_to_explain:
        :param X_importances:
        :param exclude_neighbourhood:
        :param use_parity:
        :param parity_strategy:
        :param inverse_sampling:
        :param class_names:
        :param discount_importance:
        :param uncertain_entropy_evaluator:
        :param beta:
        :param representative:
        :param density_sampling:
        :param radius_sampling:
        :param oversampling:
        :param categorical:
        :param prune:
        :param oblique:
        :param n_jobs:
        :return:
        """
        if class_names is None:
            class_names = np.unique(y)
        if class_names is not None and len(class_names) != len(np.unique(y)):
            raise ValueError('Length of class_names not aligned with number of classess in y')

        self.attributes_names = X.columns
        self.categorical = categorical

        if isinstance(X_importances, np.ndarray):
            X_importances = pd.DataFrame(X_importances, columns=self.attributes_names)

        if isinstance(instance_to_explain, (list, np.ndarray)):
            if isinstance(instance_to_explain, (list)):
                instance_to_explain = np.array([instance_to_explain])
            if len(instance_to_explain.shape) == 2:
                return self.fit_bounding_boxes(X=X, y=y, boundiong_box_points=instance_to_explain,
                                               X_importances=X_importances, exclude_neighbourhood=exclude_neighbourhood,
                                               use_parity=use_parity, inverse_sampling=inverse_sampling,
                                               class_names=class_names, parity_strategy=parity_strategy,
                                               radius_sampling=radius_sampling, discount_importance=discount_importance,
                                               uncertain_entropy_evaluator=uncertain_entropy_evaluator, beta=beta,
                                               representative=representative, density_sampling=density_sampling,
                                               oversampling=oversampling, categorical=categorical, prune=prune,
                                               oblique=oblique, n_jobs=n_jobs)
            else:
                raise ValueError('Dimensions of point to explain not aligned with dataset')

    def fit_bounding_boxes(self, X, y, boundiong_box_points, X_importances=None, exclude_neighbourhood=False,
                           use_parity=True, parity_strategy='global', inverse_sampling=False, class_names=None,
                           discount_importance=False, uncertain_entropy_evaluator=UncertainEntropyEvaluator(), beta=1,
                           representative='centroid', density_sampling=False, radius_sampling=False, oversampling=False,
                           categorical=None, prune=False, oblique=False, n_jobs=None):
        """

        :param X:
        :param y:
        :param boundiong_box_points:
        :param X_importances:
        :param exclude_neighbourhood:
        :param use_parity:
        :param parity_strategy:
        :param inverse_sampling:
        :param class_names:
        :param discount_importance:
        :param uncertain_entropy_evaluator:
        :param beta:
        :param representative:
        :param density_sampling:
        :param radius_sampling:
        :param oversampling:
        :param categorical:
        :param prune:
        :param oblique:
        :param n_jobs:
        """
        if class_names is None:
            class_names = np.unique(y)
        if class_names is not None and len(class_names) != len(np.unique(y)):
            raise ValueError('Length of class_names not aligned with number of classess in y')

        if isinstance(boundiong_box_points, (list)):
            boundiong_box_points = np.array(boundiong_box_points)
        if len(boundiong_box_points.shape) != 2:
            raise ValueError('Bounding box should be 2D.')

        if X_importances is not None:
            if self.classifier is not None:
                warnings.warn(
                    "WARNING: when classifier is provided, X_importances and discount_importance have no effect.")
            if not isinstance(X_importances, pd.DataFrame):
                raise ValueError('Feature importance matrix has to be DataFrame.')

        X_train_sample, X_train_sample_importances = self.create_sample_bb(X, np.argmax(self.predict_proba(X), axis=1),
                                                                           boundiong_box_points,
                                                                           X_importances=X_importances,
                                                                           exclude_neighbourhood=exclude_neighbourhood,
                                                                           use_parity=use_parity,
                                                                           inverse_sampling=inverse_sampling,
                                                                           class_names=class_names,
                                                                           representative=representative,
                                                                           density_sampling=density_sampling,
                                                                           radius_sampling=radius_sampling,
                                                                           n_jobs=n_jobs,
                                                                           parity_strategy=parity_strategy,
                                                                           oversampling=oversampling,
                                                                           categorical=categorical)
        y_train_sample = self.predict_proba(X_train_sample)
        # limit features here

        threshold_proba = np.max(self.predict_proba(boundiong_box_points))
        proball = np.max(self.predict_proba(X_train_sample), axis=1)
        threshold = np.min((np.mean(proball) - self.uncertainty_sigma * np.std(proball), threshold_proba))
        X_train_sample = X_train_sample[proball >= threshold]

        # no proba predictor
        y_train_sample_proba = self.predict_proba(X_train_sample)
        hot = np.argmax(y_train_sample_proba, axis=1)
        y_train_sample = np.zeros(y_train_sample_proba.shape)
        for i in range(0, len(y_train_sample)):
            y_train_sample[i, hot[i]] = 1

        uarff = LUX.generate_uarff(X_train_sample, y_train_sample, X_importances=X_train_sample_importances,
                                   categorical=categorical, class_names=class_names)
        self.data = Data.parse_uarff_from_string(uarff)

        self.uid3 = UId3(max_depth=self.max_depth, node_size_limit=self.node_size_limit,
                         grow_confidence_threshold=self.grow_confidence_threshold,
                         min_impurity_decrease=self.min_impurity_decrease)
        self.uid3.PARALLEL_ENTRY_FACTOR = 100
        if self.classifier is not None:
            self.tree = self.uid3.fit(self.data, entropyEvaluator=uncertain_entropy_evaluator,
                                      classifier=self.classifier, depth=0, beta=beta, prune=prune, oblique=oblique,
                                      discount_importance=discount_importance, n_jobs=n_jobs)
        else:
            self.tree = self.uid3.fit(self.data, entropyEvaluator=uncertain_entropy_evaluator, depth=0,
                                      discount_importance=discount_importance, beta=beta, prune=prune, oblique=oblique,
                                      n_jobs=n_jobs)

    def create_sample_bb(self, X, y, boundiong_box_points, X_importances=None, exclude_neighbourhood=False,
                         use_parity=True, parity_strategy='global', inverse_sampling=False, class_names=None,
                         representative='centroid', density_sampling=False, radius_sampling=False, radius=None,
                         oversampling=False, categorical=None, n_jobs=None):
        """

        :param X:
        :param y:
        :param boundiong_box_points:
        :param X_importances:
        :param exclude_neighbourhood:
        :param use_parity:
        :param parity_strategy:
        :param inverse_sampling:
        :param class_names:
        :param representative:
        :param density_sampling:
        :param radius_sampling:
        :param radius:
        :param oversampling:
        :param categorical:
        :param n_jobs:
        :return:
        """
        neighbourhoods = []
        importances = []

        if X_importances is not None:
            if not isinstance(X_importances, pd.DataFrame):
                raise ValueError('Feature importance matrix has to be DataFrame.')

        if categorical is None or sum(categorical) == 0:
            metric = 'minkowski'
        else:
            metric = 'precomputed'

        if use_parity:
            for instance_to_explain in boundiong_box_points:
                nn_instance_to_explain = np.array(instance_to_explain).reshape(1, -1)
                instance_class = np.argmax(self.predict_proba(np.array(instance_to_explain).reshape(1, -1)))
                class_names_instance_last = [c for c in class_names if c not in [instance_class]] + [instance_class]
                neighbourhoods_bbox = []
                importances_bbox = []
                for c in class_names_instance_last:
                    X_c_only = X[y == c]
                    if self.neighborhood_size <= 1.0:
                        n_neighbors = min(len(X_c_only) - 1, max(1, int(self.neighborhood_size * len(X_c_only))))
                        # TODO ADD WARNING
                        nn = NearestNeighbors(n_neighbors=max(1, int(n_neighbors / len(boundiong_box_points))),
                                              n_jobs=n_jobs)
                    else:
                        min_occurances_lables = list(np.array(y)).count(c)
                        if self.neighborhood_size > min_occurances_lables:
                            n_neighbors = min_occurances_lables
                            warnings.warn(
                                "WARNING: neighbourhood size select is smaller than number of instances within a class.")
                            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
                        else:
                            nn = NearestNeighbors(n_neighbors=self.neighborhood_size, n_jobs=n_jobs)

                    if inverse_sampling and c == instance_class:
                        neighbourhoods_bbox_inv, importances_bbox_inv = self.__inverse_sampling(X, y,
                                                                                                instance_to_explain=instance_to_explain,
                                                                                                sampling_class_label=instance_class,
                                                                                                opposite_neighbourhood=neighbourhoods_bbox,
                                                                                                X_importances=X_importances,
                                                                                                representative=representative,
                                                                                                categorical=categorical,
                                                                                                metric=metric,
                                                                                                nn=nn, n_jobs=n_jobs)
                        neighbourhoods_bbox += neighbourhoods_bbox_inv
                        if X_importances is not None:
                            importances_bbox += importances_bbox_inv

                    if metric == 'precomputed':
                        ids_c = \
                            gower.gower_topn(nn_instance_to_explain, X_c_only, cat_features=categorical,
                                             n=nn.n_neighbors,
                                             n_jobs=n_jobs)['index']
                    else:
                        nn.fit(X_c_only.values)
                        _, ids_c = nn.kneighbors(nn_instance_to_explain)
                    neighbourhoods_bbox.append(X_c_only.iloc[ids_c.ravel()])
                    if X_importances is not None:
                        X_c_only_importances = X_importances.loc[(y == c)]
                        neighbourhood_importances = X_c_only_importances.iloc[ids_c.ravel()]
                        importances_bbox.append(neighbourhood_importances)

                neighbourhoods += neighbourhoods_bbox
                if X_importances is not None:
                    importances += importances_bbox

            X_train_sample = pd.concat(neighbourhoods)
            X_train_sample = X_train_sample[~X_train_sample.index.duplicated(keep='first')]

            if X_importances is not None:
                X_train_sample_importances = pd.concat(importances)
                X_train_sample_importances = X_train_sample_importances[
                    ~X_train_sample_importances.index.duplicated(keep='first')]

            #########################################
            if parity_strategy == 'local':
                X_train_sample_c = X_train_sample.copy()
                attributes = [a for a in X_train_sample]
                X_train_sample_c['target'] = np.argmax(self.predict_proba(X_train_sample_c), axis=1)
                representations = X_train_sample_c.groupby('target').agg(np.median)
                representations['target'] = np.argmax(self.predict_proba(representations[attributes]))
                prototypes = representations[
                    representations['target'] != np.argmax(self.predict_proba(nn_instance_to_explain), axis=1)[0]][
                    attributes]
                # find distances from instance to representatives
                prototypes['distances'] = sklearn.metrics.pairwise_distances(prototypes[attributes],
                                                                             Y=nn_instance_to_explain)
                target_radius = prototypes.reset_index().iloc[prototypes['distances'].argmin()]['target']

                X_train_sample_c['distances'] = sklearn.metrics.pairwise_distances(X_train_sample_c[attributes],
                                                                                   Y=nn_instance_to_explain)
                t = X_train_sample_c[X_train_sample_c['target'] == target_radius].max()
                X_train_sample = X_train_sample_c[X_train_sample_c['distances'] <= t[0]]
                if X_importances is not None:
                    X_train_sample_importances = X_train_sample_importances[
                        (X_train_sample_c['distances'] <= t[0]).values]
                    #########################################
        else:
            if inverse_sampling:
                warnings.warn("WARNING: inverse sampling with use_parity set to False has no effect.")
            X_c_only = X
            if self.neighborhood_size <= 1.0:
                n_neighbors = min(len(X_c_only) - 1, max(1, int(self.neighborhood_size * len(X_c_only))))
                nn = NearestNeighbors(n_neighbors=max(1, int(n_neighbors / len(boundiong_box_points))), n_jobs=n_jobs,
                                      metric=metric)
            else:
                nn = NearestNeighbors(n_neighbors=self.neighborhood_size, n_jobs=n_jobs, metric=metric)

            if metric != 'precomputed':
                nn.fit(X_c_only.values)
            for instance_to_explain in boundiong_box_points:
                nn_instance_to_explain = np.array(instance_to_explain).reshape(1, -1)
                if metric == 'precomputed':
                    ids_c = \
                        gower.gower_topn(nn_instance_to_explain, X_c_only, cat_features=categorical, n=nn.n_neighbors,
                                         n_jobs=n_jobs)['index']
                else:
                    _, ids_c = nn.kneighbors(nn_instance_to_explain)
                neighbourhoods.append(X_c_only.iloc[ids_c.ravel()])
                if X_importances is not None:
                    neighbourhood_importances = X_importances.iloc[ids_c.ravel()]

            X_train_sample = X_c_only[X_c_only.index.isin(pd.concat(neighbourhoods).index)]
            if X_importances is not None:
                X_train_sample_importances = X_importances[X_importances.index.isin(neighbourhood_importances.index)]

        if density_sampling:
            X_copy = X.copy()
            # X_copy_full = X.copy()
            # for class_in_consideration in np.unique(y):
            #    X_copy = X_copy_full[y==class_in_consideration]

            clu = OPTICS(min_samples=self.min_samples, metric=metric, n_jobs=n_jobs)
            if metric == 'precomputed':
                optics_input = gower.gower_matrix(X_copy.iloc[:, ], cat_features=categorical, n_jobs=n_jobs)
                labels = clu.fit_predict(optics_input)
            else:
                labels = clu.fit_predict(X_copy)
            X_copy['label'] = labels

            X_train_sample['label'] = X_copy['label']
            # remove noise?
            # X_train_sample=X_train_sample[X_train_sample['label']!=-1] #REOVIN NOISE
            labels_to_add = X_copy[X_copy.index.isin(X_train_sample.index)]['label'].unique()
            labels_to_add = labels_to_add[labels_to_add != -1]

            total = pd.concat((X_train_sample, X.loc[X_copy[X_copy['label'].isin(labels_to_add)].index]))
            X_train_sample = total[~total.index.duplicated(keep='first')].drop(columns=['label'])
            if X_importances is not None:
                X_importances_copy = X_importances.copy()
                X_importances_copy['label'] = X_copy['label'].values
                total_importances = pd.concat(
                    (X_train_sample_importances, X_importances_copy[X_importances_copy['label'].isin(labels_to_add)]))
                X_train_sample_importances = total[~total.index.duplicated(keep='first')].drop(columns=['label'])

        if radius_sampling:
            instance_to_explain = boundiong_box_points[
                0]  # Todo in case of BBozes, rasius should be calculated for all of them
            X_train_sample = X.loc[X_train_sample.index]
            if radius is None:
                if metric == 'precomputed':
                    distances = gower.gower_matrix(np.array(instance_to_explain).reshape(1, -1),
                                                   X_train_sample.iloc[:, ], cat_features=categorical, n_jobs=n_jobs)
                else:
                    distances = sklearn.metrics.pairwise_distances(X_train_sample, instance_to_explain.reshape(1, -1))
                radius = max(distances)

            if metric == 'precomputed':
                distances = gower.gower_matrix(np.array(instance_to_explain).reshape(1, -1), X_train_sample.iloc[:, ],
                                               cat_features=categorical, n_jobs=n_jobs)
            else:
                distances = sklearn.metrics.pairwise_distances(X, instance_to_explain.reshape(1, -1))
            idxs, _ = np.where(distances <= radius)
            X_train_sample = X.iloc[idxs]
            if X_importances is not None:
                X_train_sample_importances = X_importances.iloc[idxs]

        if exclude_neighbourhood:
            X_train_sample = X.loc[~X_train_sample.index]
            if X_importances is not None:
                X_train_sample_importances = X_importances.loc[~X_train_sample_importances.index]

        if oversampling:
            if X_importances is not None:
                warnings.warn("WARNING: X_importances have no effect when oversampling is True.")
                X_importances = None
            if self.oversampling_strategy == self.OS_STRATEGY_SMOTE:
                instance_to_explain = boundiong_box_points[
                    0]  # Todo in case of BBozes, rasius should be calculated for all of them
                X_train_sample = self.__oversample_smote(X_train_sample, categorical=categorical,
                                                         instance_to_explain=instance_to_explain)
            elif self.oversampling_strategy == self.OS_STRATEGY_IMPORTANCE:
                instance_to_explain = boundiong_box_points[0]
                X_train_sample = self.__importance_sampler(X_train_sample, instance_to_explain)
            elif self.oversampling_strategy == self.OS_STRATEGY_BOTH:
                instance_to_explain = boundiong_box_points[0]
                X_train_sample = self.__importance_sampler(X_train_sample, instance_to_explain)
                X_train_sample = self.__oversample_smote(X_train_sample, categorical=categorical,
                                                         instance_to_explain=instance_to_explain)

        if X_importances is not None:
            return X_train_sample, X_train_sample_importances
        else:
            return X_train_sample, None

    def __oversample_smote(self, X_train_sample, sigma=1, iterations=1, instance_to_explain=None, categorical=None):
        """

        :param X_train_sample:
        :param sigma:
        :param iterations:
        :param instance_to_explain:
        :param categorical:
        :return:
        """
        for iteration in np.arange(0, iterations):
            try:
                sm = UncertainSMOTE(predict_proba=self.predict_proba, sigma=sigma, sampling_strategy='all',
                                    min_samples=self.min_generate_samples,
                                    instance_to_explain=instance_to_explain)
                X_train_sample, _ = sm.fit_resample(X_train_sample,
                                                    np.argmax(self.predict_proba(X_train_sample), axis=1))
            except:
                warnings.warn("WARNING: Selected class has low number of borderline points.")

        return X_train_sample

    def __inverse_sampling(self, X, y, instance_to_explain, nn, sampling_class_label, opposite_neighbourhood,
                           X_importances=None, representative='centroid', categorical=None, metric='minkowski',
                           n_jobs=None):
        """

        :param X:
        :param y:
        :param instance_to_explain:
        :param nn:
        :param sampling_class_label:
        :param opposite_neighbourhood:
        :param X_importances:
        :param representative:
        :param categorical:
        :param metric:
        :param n_jobs:
        :return:
        """
        # representative as centropid (mean value), but cna be prototype, nearest, etc.
        X_sample = X[y == sampling_class_label]
        if X_importances is not None:
            X_importances_sample = X_importances[(y == sampling_class_label).values]

        nn_instance_to_explain = np.array(instance_to_explain).reshape(1, -1)

        inverse_neighbourhood = []
        inverse_neighbourhood_importances = []
        for data in opposite_neighbourhood:
            # from this class, select representative
            if representative == self.REPRESENTATIVE_CENTROID:
                representative_sample = data.mean(axis=0)
            elif representative == self.REPRESENTATIVE_NEAREST:
                # find nearest example to explain_instance and use it as representative_sample
                if metric == 'precomputed':
                    ids = gower.gower_topn(nn_instance_to_explain, data, n=1, cat_features=categorical, n_jobs=n_jobs)[
                        'index']
                    representative_sample = data.iloc[ids.ravel()[0]]
                else:
                    nn_inverse = NearestNeighbors(n_neighbors=1, metric=metric)
                    nn_inverse.fit(data)
                    _, ids = nn_inverse.kneighbors(nn_instance_to_explain)
                    representative_sample = data.iloc[ids.ravel()[
                        0]]

            # Find closest to the representative sample
            if metric == 'precomputed':
                ids_c = gower.gower_topn(np.array(representative_sample).reshape(1, -1), X_sample, n=nn.n_neighbors,
                                         cat_features=categorical, n_jobs=n_jobs)['index']
            else:
                nn.fit(X_sample)
                _, ids_c = nn.kneighbors(np.array(representative_sample).reshape(1, -1))
            # Save in neighbouirhood and importances
            inverse_neighbourhood.append(X_sample.iloc[ids_c.ravel()])
            if X_importances is not None:
                inverse_neighbourhood_importances.append(X_importances_sample.iloc[ids_c.ravel()])

        if X_importances is not None:
            return inverse_neighbourhood, inverse_neighbourhood_importances
        else:
            return inverse_neighbourhood, None

    def predict(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        if isinstance(X, pd.DataFrame):
            pass
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.attributes_names)
        else:
            raise ValueError("Only 2D arrrays are allowed as an input")

        if y is None:
            y = pd.Series(np.arange(X.shape[0]), name='target_unused',
                          index=X.index)  # This is not used, but Data resered last

        X = pd.concat((X, y), axis=1)
        XData = Data.parse_dataframe(X, 'lux')
        return [int(f.get_name()) for f in self.uid3.predict(XData.get_instances())]

    def justify(self, X, to_dict=False, reduce=True):
        """Traverse down the path for given x.
        :param X:
        :param to_dict:
        :param reduce:
        :return:
        """
        if isinstance(X, pd.DataFrame):
            pass
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.attributes_names)
        else:
            raise ValueError("Only 2D arrrays are allowed as an input")

        y = pd.Series(np.arange(X.shape[0]), name='target_unused',
                      index=X.index)  # This is not used, but Data resered last
        X = pd.concat((X, y), axis=1)
        XData = Data.parse_dataframe(X, 'lux')

        if to_dict:
            return [self.uid3.tree.justification_tree(i).to_dict(reduce=reduce) for i in XData.get_instances()]
        else:
            return [self.uid3.tree.justification_tree(i).to_pseudocode(reduce=reduce) for i in XData.get_instances()]

    def __get_covered(self, rule, dataset, features, categorical=None):
        """

        :param rule:
        :param dataset:
        :param features:
        :param categorical:
        :return:
        """
        if categorical is None:
            categorical = [False] * len(features)
        query = []
        if rule == {}:
            return 0, 0
        for i, v in rule.items():
            op = '' if dict(zip(features, categorical))[i] == False else '=='
            query.append(f'{i}{op}' + f'and {i}{op}'.join(v))

        covered = dataset.query(' and '.join(query))
        return covered

    def counterfactual(self, instance_to_explain, background, counterfactual_representative='medoid', reduce=True,
                       topn=None, n_jobs=None):
        """

        :param instance_to_explain:
        :param background:
        :param counterfactual_representative:
        :param reduce:
        :param topn:
        :param n_jobs:
        :return:
        """
        not_class = np.argmax(self.predict_proba(instance_to_explain))
        rules = self.uid3.tree.to_dict(reduce=reduce)
        bbox_predictions = np.argmax(self.predict_proba(background), axis=1)
        lux_predictions = self.predict(background)
        background = background[(bbox_predictions == lux_predictions)]
        # filter out rules with class same as not_class
        counterfactual_rules = []
        for rule in rules:
            if int(rule['prediction']) != not_class:
                # find coverage points from background
                rule['covered'] = self.__get_covered(rule['rule'], background, self.attributes_names, self.categorical)
                if len(rule['covered']) == 0:
                    continue

                counterfactual_rules.append(rule)

                # find candidates from background according to counterfactual_representative
                if counterfactual_representative == self.CF_REPRESENTATIVE_MEDOID:
                    if self.categorical is not None:
                        pass
                    else:
                        distances = sklearn.metrics.pairwise_distances(rule['covered'])
                        ids = np.argmin(distances.sum(axis=0))
                        dist = sklearn.metrics.pairwise_distances(rule['covered'].iloc[ids].values.reshape(1, -1),
                                                                  instance_to_explain)
                        representative_sample = rule['covered'].iloc[ids]
                        rule['counterfactual'] = representative_sample
                        rule['distance'] = dist
                elif counterfactual_representative == self.CF_REPRESENTATIVE_NEAREST:
                    if self.categorical is not None:
                        ids_dist = gower.gower_topn(instance_to_explain, rule['covered'], n=1,
                                                    cat_features=self.categorical, n_jobs=n_jobs)
                        representative_sample = rule['covered'].iloc[ids_dist['index'].ravel()[0]]
                        rule['counterfactual'] = representative_sample
                        dist = ids_dist['values']
                        rule['distance'] = dist
                    else:
                        nn_inverse = NearestNeighbors(n_neighbors=1, metric='minkowski')
                        nn_inverse.fit(rule['covered'])
                        dist, ids = nn_inverse.kneighbors(instance_to_explain)
                        representative_sample = rule['covered'].iloc[ids.ravel()[0]]
                        rule['counterfactual'] = representative_sample
                        rule['distance'] = dist
                else:
                    raise ValueError("Counterfactual representative can be either 'medoid' or 'nearest'")

        # find closest representative to the instance_to_explain and return as counterfactual, along with rules
        counterfactual_rules = sorted(counterfactual_rules, key=lambda d: d['distance'])
        if topn is None:
            return counterfactual_rules
        else:
            return counterfactual_rules[:topn]

    def __getshap(self, X_train_sample):
        """

        :param X_train_sample:
        :return:
        """
        # calculate shap values
        try:
            explainer = shap.Explainer(self.classifier, X_train_sample)
            if hasattr(explainer, "shap_values"):
                shap_values = explainer.shap_values(X_train_sample, check_additivity=False)
            else:
                shap_values = explainer(X_train_sample).values
                shap_values = [sv for sv in np.moveaxis(shap_values, 2, 0)]
            if hasattr(explainer, "expected_value"):
                expected_values = explainer.expected_value
            else:
                expected_values = [np.mean(v) for v in shap_values]
        except TypeError:
            explainer = shap.Explainer(self.predict_proba, X_train_sample)
            shap_values = explainer(X_train_sample).values
            shap_values = [sv for sv in np.moveaxis(shap_values, 2, 0)]
            expected_values = [np.mean(v) for v in shap_values]

        if type(shap_values) is not list:
            shap_values = [-shap_values, shap_values]
            expected_values = [np.mean(v) for v in shap_values]

        return shap_values, expected_values

    def __importance_sampler(self, X_train_sample, instance_to_explain, num=10):
        """

        :param X_train_sample:
        :param instance_to_explain:
        :param num:
        :return:
        """
        shap_values, _ = self.__getshap(X_train_sample)
        abs_shap = np.array([abs(sv).mean(1) for sv in shap_values])
        indexer = self.classifier.predict(X_train_sample)
        shapclass = []

        for i in range(0, len(X_train_sample)):
            # we move sample towards the expected value, which should be decision boundary in balanced, binary case
            best_index = indexer[i]
            shapclass.append([shap_values[best_index][i, :]])
        shapclass = np.concatenate(shapclass)
        shapcols = [c + '_shap' for c in X_train_sample.columns]
        cols = [c for c in X_train_sample.columns]

        shapdf = pd.DataFrame(shapclass, columns=shapcols)

        fulldf = pd.concat([X_train_sample.reset_index(drop=True), shapdf.reset_index(drop=True)], axis=1)
        fulldf.index = X_train_sample.index
        class_of_i2e = self.classifier.predict(instance_to_explain.reshape(1, -1))
        predictions = self.classifier.predict(fulldf[cols])
        # fulldf=fulldf[predictions==class_of_i2e]

        gradsf = {}
        gradst = []

        for cl in np.unique(indexer):
            gradcl = []
            gradstcl = []
            for dim in range(0, X_train_sample.shape[1]):
                mask = indexer == cl
                xs = X_train_sample.iloc[mask, dim]
                ys = shapclass[mask, dim]
                # plt.plot(xs,ys)
                # plt.show()
                grads = np.gradient(ys, xs)
                gradstcl.append(grads)
                svrr = LinearRegression()  # SVR()
                svrr.fit(xs.values.reshape(-1, 1), ys)

                F = lambda x, svr=svrr: svr.predict(x.reshape(1, -1))
                gradient = nd.Gradient(F)

                gradcl.append(gradient)
            gradsf[cl] = gradcl
            gradst.append(gradstcl)

        alpha = np.ones(len(cols)) * (shapclass.max() - shapclass.min())

        def perturb(x, num, alpha, gradients, cols, shapcols):
            newx = []
            last = x[cols].values
            newx.append(last)
            cl = self.classifier.predict(last.reshape(1, -1))[0]

            grad = np.array([g(last[i]) for i, g in enumerate(gradients[cl])])
            for _ in range(0, num):
                last = last - alpha * grad
                if cl != self.classifier.predict(last.reshape(1, -1))[0]:
                    break
                newx.append(last)
            return np.array(newx)

        if fulldf.shape[0] > 0:
            upsamples = np.concatenate(
                fulldf.sample(int(self.min_generate_samples * len(fulldf))).apply(perturb, args=(
                num, alpha, gradsf, cols, shapcols),
                                                                                  axis=1).values)
            upsamples = upsamples[
                        np.random.choice(upsamples.shape[0], max(len(fulldf), upsamples.shape[0]), replace=False), :]
        else:
            upsamples = fulldf

        return pd.concat((pd.DataFrame(upsamples, columns=X_train_sample.columns), X_train_sample))

    def to_HMR(self):
        """

        :return:
        """
        return self.tree.to_HMR()

    @staticmethod
    def generate_uarff(X, y, class_names, X_importances=None, categorical=None):
        """ Generates uncertain ARFF file
            :param X: DataFrame containing dataset for training
            :param y: target values returned by predict_proba function
            :param class_names: names for the classes to be used in uID3
            :param X_importances: confidence for each reading obtained. This matrix should be normalized to the range [0;1].
            :param categorical: array indicating which parameters should be treated as categorical
            :return: String representing the ARFF file

        """
        if X_importances is not None:
            if not isinstance(X_importances, pd.DataFrame):
                raise ValueError('Reading confidence matrix has to be DataFrame.')
            if X.shape != X_importances.shape:
                raise ValueError("Confidence for readings have to be exaclty the size of X.")
        if categorical is None:
            categorical = [False] * X.shape[1]

        uarff = "@relation lux\n\n"
        for i, (f, t) in enumerate(zip(X.columns, X.dtypes)):
            if t in (int, float, np.int32, np.int64) and not categorical[i]:
                uarff += f'@attribute {f} @REAL\n'
            elif categorical[i]:
                domain = ','.join(map(str, list(X[f].unique())))
                uarff += '@attribute ' + f + ' {' + domain + '}\n'

        domain = ','.join([str(cn) for cn in class_names])
        uarff += '@attribute class {' + domain + '}\n'

        uarff += '@data\n'
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                if X_importances is not None:
                    uarff += f'{X.iloc[i, j]}[{X_importances.iloc[i, j]}],'
                else:
                    uarff += f'{X.iloc[i, j]}[1],'
            uarff += ';'.join([f'{c}[{p}]' for c, p in zip(class_names, y[i, :])]) + '\n'
        return uarff
