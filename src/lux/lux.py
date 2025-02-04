from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import pandas as pd

from lux.samplers import UncertainSMOTE
from lux.pyuid3.data import Data
from lux.pyuid3.entropy_evaluator import UncertainEntropyEvaluator
from lux.pyuid3.uid3 import UId3
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS
import re
import shap
import sklearn
import gower_multiprocessing as gower
import numpy as np
import warnings
import inspect
import pandas.api.types as ptypes

from lux.samplers import ImportanceSampler


class LUX(BaseEstimator):
    """
    This class contains functions that implement generation of local rule-based model-agnostic explanations.
    (np.array(iris_instance))
    """

    REPRESENTATIVE_CENTROID = "centroid"
    "REPRESENTATIVE_CENTROID (:obj:`str`): A constant representing centroid as the representative strategy."

    REPRESENTATIVE_NEAREST = "nearest"
    "REPRESENTATIVE_NEAREST (:obj:`str`): A constant representing nearest as the representative strategy."

    CF_REPRESENTATIVE_MEDOID = "medoid"
    "CF_REPRESENTATIVE_MEDOID (:obj:`str`): A constant representing medoid as the counterfactual representative strategy."

    CF_REPRESENTATIVE_NEAREST = "nearest"
    "CF_REPRESENTATIVE_NEAREST (:obj:`str`): A constant representing nearest as the counterfactual representative strategy."

    OS_STRATEGY_SMOTE = 'smote'
    "OS_STRATEGY_SMOTE (:obj:`str`): A constant representing SMOTE as the oversampling strategy."

    OS_STRATEGY_IMPORTANCE = 'importance'
    "OS_STRATEGY_IMPORTANCE (:obj:`str`): A constant representing importance sampling as the oversampling strategy."

    OS_STRATEGY_BOTH = 'both'
    "OS_STRATEGY_BOTH (:obj:`str`): A constant representing both SMOTE and importance sampling as the oversampling strategy."

    def __init__(self, predict_proba, classifier=None, neighborhood_size=0.1, max_depth=None, node_size_limit=1,
                 grow_confidence_threshold=0, min_impurity_decrease=0, min_samples=5, min_generate_samples=0.02,
                 uncertainty_sigma=2, oversampling_strategy='both'):
        """ Initialize the LUX explainer model.

        :param predict_proba: callable
            The predict_proba function of the balckbox classifier.
        :type predict_proba: callable
        :param classifier: object, optional
            The underlying classifier. If it is provided the SHAP-based sampling can be used.
        :param neighborhood_size: float, optional
            The neighborhood size for generating explanations. Default is 0.1.
        :type neighborhood_size: float
        :param max_depth: int, optional
            The maximum depth of the decision tree. Default is None meaning no limit.
        :type max_depth: int
        :param node_size_limit: int, optional
            The minimum number of samples required to split an internal node. Default is 1.
        :type node_size_limit: int
        :param grow_confidence_threshold: float, optional
            The threshold for growing decision tree nodes. Default is 0.
        :type grow_confidence_threshold: float
        :param min_impurity_decrease: float, optional
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            Default is 0.
        :type min_impurity_decrease: float
        :param min_samples: int, optional
            The minimum number of samples required to be at a leaf node. Default is 5.
        :type min_samples: int
        :param min_generate_samples: float, optional
            The minimum proportion of the dataset size to generate perturbed instances. This is used by the UncertainSMOTE algotrothm. Default is 0.02.
        :type min_generate_samples: float
        :param uncertainty_sigma: float, optional
            The uncertainty parameter sigma used in the filtering uncertain samples. Every sample that is 2*uncertainty_sigma away from the mean will be removed. Default is 2.
        :type uncertainty_sigma: float
        :param oversampling_strategy: str, optional
            The strategy for oversampling. It can be 'smote', 'importance', or 'both'. Default is 'both'.
        :type oversampling_strategy: str
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

    # write a getter for classifier

    def fit(self, X, y, instance_to_explain, X_importances=None, exclude_neighbourhood=False, use_parity=True,
            parity_strategy='global', inverse_sampling=True, class_names=None, discount_importance=False,
            uncertain_entropy_evaluator=UncertainEntropyEvaluator(), beta=1, representative='centroid',
            density_sampling=False, radius_sampling=False, oversampling=True, categorical=None, prune=True,
            oblique=True, n_jobs=None):
        """ Fit the LUX explainer model.

        :param X:
            The input data used to train the model.
        :type X: pandas.DataFrame
        :param y:
            The target values corresponding to the input data.
        :type y: array-like
        :param instance_to_explain:
            The instance(s) to explain. Can be a single instance or a list/array of instances. The instances are not explained one after another, but the neighbourhood is created
            for the whole set of instances. Hence, they form so called bounding box for the explanation.
        :type instance_to_explain: array-like or list
        :param X_importances: optional
            The importances of features in the input data. If provided as a DataFrame, column names should match feature names.
        :type X_importances: array-like or pandas.DataFrame or None
        :param exclude_neighbourhood: optional
            Whether to exclude the neighborhood of the instance(s) being explained. Default is False.
        :type exclude_neighbourhood: bool
        :param use_parity: optional
            Whether to use parity constraints in explanation generation. Default is True.
        :type use_parity: bool
        :param parity_strategy: optional
            The strategy for applying parity constraints. It can be 'global' or 'local'. Default is 'global'.
        :type parity_strategy: str
        :param inverse_sampling: optional
            Whether to use inverse sampling for feature selection. Default is True.
        :type inverse_sampling: bool
        :param class_names: optional
            The names of the classes. If not provided, inferred from the target values.
        :type class_names: array-like or None
        :param discount_importance: optional
            Whether to discount feature importance. Default is False.
        :type discount_importance: bool
        :param uncertain_entropy_evaluator: optional
            The evaluator for uncertain entropy. Default is UncertainEntropyEvaluator().
        :type uncertain_entropy_evaluator: object
        :param beta: optional
            The beta parameter for F-beta score used in uncertain entropy computation. Default is 1.
        :type beta: float
        :param representative: optional
            The representative method for selecting representative instances. It can be 'centroid' or 'prototype'.
            Default is 'centroid'.
        :type representative: str
        :param density_sampling: optional
            Whether to use density-based sampling for instance selection. Default is False.
        :type density_sampling: bool
        :param radius_sampling: optional
            Whether to use radius-based sampling for instance selection. Default is False.
        :type radius_sampling: bool
        :param oversampling: optional
            Whether to perform oversampling of instances. Default is False.
        :type oversampling: bool
        :param categorical: optional
            A list indicating whether each feature is categorical or not.
        :type categorical: array-like or None
        :param prune: optional
            Whether to prune branches in decision tree that produces splits which do not change classification result. Default is True.
        :type prune: bool
        :param oblique: optional
            Whether to use oblique decision rules. Default is False.
        :type oblique: bool
        :param n_jobs: optional
            The number of parallel jobs to run. Default is None.
        :type n_jobs: int or None

        :return:
            The trained LUX explainer model.
        :rtype: lux.lux.LUX
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
                return self.fit_bounding_boxes(X=X, y=y, bounding_box_points=instance_to_explain,
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

    def fit_bounding_boxes(self, X, y, bounding_box_points, X_importances=None, exclude_neighbourhood=False,
                           use_parity=True, parity_strategy='global', inverse_sampling=False, class_names=None,
                           discount_importance=False, uncertain_entropy_evaluator=UncertainEntropyEvaluator(), beta=1,
                           representative='centroid', density_sampling=False, radius_sampling=False, oversampling=False,
                           categorical=None, prune=False, oblique=False, n_jobs=None):
        """ Fit LUX explainer model for the neighbourhood data defined by the bounding box constructed of several points.
        Usually only one point is provided.

        :param X:
            Input features.
        :type X: array-like or sparse matrix of shape (n_samples, n_features)
        :param y:
            Target values.
        :param bounding_box_points:
            Points defining the bounding box.
        :type bounding_box_points: array-like of shape (n_points, n_dimensions)
        :param X_importances:
            Importance matrix for features. Default is None.
        :param exclude_neighbourhood:
            Whether to exclude neighborhood points. Default is False.
        :param use_parity:
            Whether to use parity. Default is True.
        :param parity_strategy:
            Strategy for parity. Default is 'global'.
        :param inverse_sampling:
            Whether to use inverse sampling. Default is False.
        :param class_names:
            Names of classes. Default is None.
        :param discount_importance:
            Whether to discount importance. Default is False.
        :param uncertain_entropy_evaluator:
            Evaluator for uncertain entropy. Default is UncertainEntropyEvaluator().
        :param beta:
            Beta value for fitting. Default is 1.
        :param representative:
            Representative strategy. Default is 'centroid'.
        :param density_sampling:
            Whether to use density sampling. Default is False.
        :param radius_sampling:
            Whether to use radius sampling. Default is False.
        :param oversampling:
            Whether to use oversampling. Default is False.
        :param categorical:
            Categorical information. Default is None.
        :param prune:
            Whether to prune. Default is False.
        :param oblique:
            Whether to use oblique splits. Default is False.
        :param n_jobs:
            Number of jobs to run in parallel. Default is None.

        Raises:
        :raises ValueError:
            If the length of class_names does not match the number of classes in y,
                           or if bounding_box_points is not 2D.
        """
        if class_names is None:
            class_names = np.unique(y)
        if class_names is not None and len(class_names) != len(np.unique(y)):
            raise ValueError('Length of class_names not aligned with number of classes in y')

        if isinstance(bounding_box_points, (list)):
            bounding_box_points = np.array(bounding_box_points)
        if len(bounding_box_points.shape) != 2:
            raise ValueError('Bounding box should be 2D.')

        if X_importances is not None:
            if self.classifier is not None:
                warnings.warn(
                    "WARNING: when classifier is provided, X_importances and discount_importance have no effect.")
            if not isinstance(X_importances, pd.DataFrame):
                raise ValueError('Feature importance matrix has to be DataFrame.')

        X_train_sample, X_train_sample_importances = self.create_sample_bb(X, np.argmax(
            self.predict_proba(self.process_input(X)), axis=1),
                                                                           bounding_box_points,
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
        #y_train_sample = self.predict_proba(self.process_input(X_train_sample))
        # limit features here

        # threshold_proba = np.max(self.predict_proba(boundiong_box_points))
        # proball = np.max(self.predict_proba(X_train_sample), axis=1)
        # threshold = np.min((np.mean(proball) - self.uncertainty_sigma * np.std(proball), threshold_proba))
        # X_train_sample = X_train_sample[proball >= threshold]

        # no proba predictor
        y_train_sample_proba = self.predict_proba(self.process_input(X_train_sample))
        hot = np.argmax(y_train_sample_proba, axis=1)
        y_train_sample = np.zeros(y_train_sample_proba.shape)
        for i in range(0, len(y_train_sample)):
            y_train_sample[i, hot[i]] = 1

        X_train_sample['class'] = pd.Series(hot, index=X_train_sample.index)

        self.data = Data.parse_dataframe(X_train_sample, X_train_sample_importances)

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
        return self

    def create_sample_bb(self, X, y, bounding_box_points, X_importances=None, exclude_neighbourhood=False,
                         use_parity=True, parity_strategy='global', inverse_sampling=False, class_names=None,
                         representative='centroid', density_sampling=False, radius_sampling=False, radius=None,
                         oversampling=False, categorical=None, n_jobs=None):
        """ Create a sample for the LUX explainer to be fitted to, based on the provided data.


        :param X:
           Input features.
        :type X: array-like or sparse matrix of shape (n_samples, n_features)
        :param y:
           Target values.
        :param bounding_box_points:
           Points defining the bounding box.
        :type bounding_box_points: array-like of shape (n_points, n_dimensions)
        :param X_importances:
           Importance matrix for features. Default is None.
        :param exclude_neighbourhood:
           Whether to exclude neighborhood points. Default is False.
        :param use_parity:
           Whether to use parity. Default is True.
        :param parity_strategy:
           Strategy for parity. Default is 'global'.
        :param inverse_sampling:
           Whether to use inverse sampling. Default is False.
        :param class_names:
           Names of classes. Default is None.
        :param representative:
           Representative strategy. Default is 'centroid'.
        :param density_sampling:
           Whether to use density sampling. Default is False.
        :param radius_sampling:
           Whether to use radius sampling. Default is False.
        :param radius:
           Radius for radius sampling. Default is None.
        :param oversampling:
           Whether to use oversampling. Default is False.
        :param categorical:
           Categorical information. Default is None.
        :param n_jobs:
           Number of jobs to run in parallel. Default is None.

        Returns:
        :return: X_train_sample:
           Sampled input features.
        :rtype: array-like or sparse matrix of shape (n_samples, n_features)
        :return: X_train_sample_importances:
           Sampled importance matrix for features.
        :rtype: pd.DataFrame or None
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

        # TODO: if classifier is present, then use it to obtain SHAP, thenm

        if use_parity:
            for instance_to_explain in bounding_box_points:
                nn_instance_to_explain = np.array(instance_to_explain).reshape(1, -1)
                instance_class = np.argmax(
                    self.predict_proba(self.process_input(np.array(instance_to_explain).reshape(1, -1))))
                class_names_instance_last = [c for c in class_names if c not in [instance_class]] + [instance_class]
                neighbourhoods_bbox = []
                importances_bbox = []
                for c in class_names_instance_last:
                    X_c_only = X[y == c]
                    if self.neighborhood_size <= 1.0:
                        n_neighbors = min(len(X_c_only) - 1, max(1, int(self.neighborhood_size * len(X_c_only))))
                        nn = NearestNeighbors(n_neighbors=max(1, int(n_neighbors / len(bounding_box_points))),
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
                        signature = inspect.signature(gower.gower_topn)
                        has_njobs = 'n_jobs' in signature.parameters
                        if has_njobs:
                            ids_c = \
                                gower.gower_topn(nn_instance_to_explain, X_c_only, cat_features=categorical,
                                                 n=nn.n_neighbors,
                                                 n_jobs=n_jobs)['index']
                        else:
                            ids_c = \
                                gower.gower_topn(nn_instance_to_explain, X_c_only, cat_features=categorical,
                                                 n=nn.n_neighbors)['index']
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
                X_train_sample_c['target'] = np.argmax(self.predict_proba(self.process_input(X_train_sample_c)), axis=1)
                representations = X_train_sample_c.groupby('target').agg(np.median)
                representations['target'] = np.argmax(
                    self.predict_proba(self.process_input(representations[attributes])))
                prototypes = representations[
                    representations['target'] !=
                    np.argmax(self.predict_proba(self.process_input(nn_instance_to_explain)), axis=1)[0]][
                    attributes]
                # find distances from instance to representatives
                prototypes['distances'] = sklearn.metrics.pairwise_distances(prototypes[attributes],
                                                                             Y=nn_instance_to_explain)
                target_radius = prototypes.reset_index().iloc[prototypes['distances'].argmin()]['target']

                X_train_sample_c['distances'] = sklearn.metrics.pairwise_distances(X_train_sample_c[attributes],
                                                                                   Y=nn_instance_to_explain)
                t = X_train_sample_c[X_train_sample_c['target'] == target_radius].max()
                X_train_sample = X_train_sample_c[X_train_sample_c['distances'] <= t[0]][X_train_sample.columns]
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
                nn = NearestNeighbors(n_neighbors=max(1, int(n_neighbors / len(bounding_box_points))), n_jobs=n_jobs,
                                      metric=metric)
            else:
                nn = NearestNeighbors(n_neighbors=self.neighborhood_size, n_jobs=n_jobs, metric=metric)

            if metric != 'precomputed':
                nn.fit(X_c_only.values)
            for instance_to_explain in bounding_box_points:
                nn_instance_to_explain = np.array(instance_to_explain).reshape(1, -1)
                if metric == 'precomputed':
                    signature = inspect.signature(gower.gower_topn)
                    has_njobs = 'n_jobs' in signature.parameters
                    if has_njobs:
                        ids_c = \
                        gower.gower_topn(nn_instance_to_explain, X_c_only, cat_features=categorical, n=nn.n_neighbors,
                                         n_jobs=n_jobs)['index']
                    else:
                        ids_c = \
                            gower.gower_topn(nn_instance_to_explain, X_c_only, cat_features=categorical,
                                             n=nn.n_neighbors)['index']
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
                signature = inspect.signature(gower.gower_topn)
                has_njobs = 'n_jobs' in signature.parameters
                if has_njobs:
                    optics_input = gower.gower_matrix(X_copy.iloc[:, ], cat_features=categorical, n_jobs=n_jobs)
                else:
                    optics_input = gower.gower_matrix(X_copy.iloc[:, ], cat_features=categorical)
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
            instance_to_explain = bounding_box_points[
                0]  # Todo in case of BBozes, rasius should be calculated for all of them
            X_train_sample = X.loc[X_train_sample.index]
            if radius is None:
                if metric == 'precomputed':
                    signature = inspect.signature(gower.gower_topn)
                    has_njobs = 'n_jobs' in signature.parameters
                    if has_njobs:
                        distances = gower.gower_matrix(np.array(instance_to_explain).reshape(1, -1),
                                                       X_train_sample.iloc[:, ], cat_features=categorical,
                                                       n_jobs=n_jobs)
                    else:
                        distances = gower.gower_matrix(np.array(instance_to_explain).reshape(1, -1),
                                                       X_train_sample.iloc[:, ], cat_features=categorical)
                else:
                    distances = sklearn.metrics.pairwise_distances(X_train_sample, instance_to_explain.reshape(1, -1))
                radius = max(distances)

            if metric == 'precomputed':
                signature = inspect.signature(gower.gower_topn)
                has_njobs = 'n_jobs' in signature.parameters
                if has_njobs:
                    distances = gower.gower_matrix(np.array(instance_to_explain).reshape(1, -1),
                                                   X_train_sample.iloc[:, ],
                                                   cat_features=categorical, n_jobs=n_jobs)
                else:
                    distances = gower.gower_matrix(np.array(instance_to_explain).reshape(1, -1),
                                                   X_train_sample.iloc[:, ],
                                                   cat_features=categorical)
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

        # in case of dim reduciton, here is where we need to go back to original feature-space
        # the return should be made based on the indices from the X_train_sample
        if oversampling:
            if X_importances is not None:
                warnings.warn("WARNING: X_importances have no effect when oversampling is True.")
                X_importances = None
            if self.oversampling_strategy == self.OS_STRATEGY_SMOTE:
                instance_to_explain = bounding_box_points[
                    0]  # Todo in case of BBozes, rasius should be calculated for all of them
                X_train_sample = self.__oversample_smote(X_train_sample, categorical=categorical,
                                                         instance_to_explain=instance_to_explain)
            elif self.oversampling_strategy == self.OS_STRATEGY_IMPORTANCE:
                instance_to_explain = bounding_box_points[0]
                isam = ImportanceSampler(classifier=self.classifier, predict_proba=self.predict_proba,
                                         indstance_to_explain=instance_to_explain,
                                         min_generate_samples=self.min_generate_samples,process_input=self.process_input,
                                         categorical=self.categorical)
                X_train_sample = isam.fit_transform(X_train_sample)
            elif self.oversampling_strategy == self.OS_STRATEGY_BOTH:
                instance_to_explain = bounding_box_points[0]
                X_train_sample = self.__oversample_smote(X_train_sample, categorical=categorical,
                                                         instance_to_explain=instance_to_explain)
                isam = ImportanceSampler(classifier=self.classifier, predict_proba=self.predict_proba,
                                         indstance_to_explain=instance_to_explain,
                                         min_generate_samples=self.min_generate_samples,process_input=self.process_input,
                                         categorical=self.categorical)
                X_train_sample = isam.fit_transform(X_train_sample)

            cols = X_train_sample.columns
            preds = np.argmax(self.predict_proba(self.process_input(X_train_sample)), axis=1)
            cl = np.argmax(self.predict_proba(self.process_input(instance_to_explain.reshape(1, -1))))
            mask = preds == cl
            # if class of interest is smaller than other classes, balance it by sampling the instance multiple times
            diff = len(preds) / len(np.unique(preds)) - sum(mask)
            # if dataset has more instances of opposite class, fill with instance2explain
            if diff > 0:
                X_train_sample_arr = np.concatenate(
                    (X_train_sample, np.ones((int(diff), X_train_sample.shape[1])) * instance_to_explain))
                X_train_sample = pd.DataFrame(X_train_sample_arr, columns=cols)

        if X_importances is not None:
            return X_train_sample, X_train_sample_importances
        else:
            return X_train_sample, None

    def process_and_predict_proba(self, X):
        """
        Process the input data and predict the probabilities.

        :param X: data that will be passed to predict_proba function after preprocessing.
        :return: probabilities, in the same format as prodict_proba
        """
        return self.predict_proba(self.process_input(X))

    def process_input(self, X):
        """
        The main goal is to change the type of categorical values, so they fit algorithms that require categories as integers
        :param X: data that will be passed to predict_proba function after preprocessing.
        :return: preprocessed data with the same dimensions as the input one, but with changed categorical columns to integers
        """

        # Convert X to a pandas DataFrame if it's a NumPy array
        if self.categorical is None:
            return X
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        elif not isinstance(X, pd.DataFrame):
            raise TypeError("X should be either a numpy array or a pandas DataFrame")

        X.columns = self.attributes_names

        # Iterate over each column and corresponding indicator
        for i, is_categorical in enumerate(self.categorical):
            if is_categorical:
                # Convert the column to int
                X[self.attributes_names[i]] = X[self.attributes_names[i]].round().astype(int)

        return X

    def __oversample_smote(self, X_train_sample, sigma=1, iterations=1, instance_to_explain=None, categorical=None):
        """ Generate samples based on their uncertainty.

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
                                    instance_to_explain=instance_to_explain,process_input=self.process_input)
                X_train_sample, _ = sm.fit_resample(X_train_sample,
                                                    np.argmax(self.predict_proba(self.process_input(X_train_sample)),
                                                              axis=1))
            except:
                warnings.warn("WARNING: Selected class has low number of borderline points.")

        return X_train_sample

    def __inverse_sampling(self, X, y, instance_to_explain, nn, sampling_class_label, opposite_neighbourhood,
                           X_importances=None, representative='centroid', categorical=None, metric='minkowski',
                           n_jobs=None):
        """ Samples instances from opposite classes making sure every class is well represented in the representative dataset.

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
                    signature = inspect.signature(gower.gower_topn)
                    has_njobs = 'n_jobs' in signature.parameters
                    if has_njobs:
                        ids = \
                        gower.gower_topn(nn_instance_to_explain, data, n=1, cat_features=categorical, n_jobs=n_jobs)[
                            'index']
                    else:
                        ids = gower.gower_topn(nn_instance_to_explain, data, n=1, cat_features=categorical)[
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
                signature = inspect.signature(gower.gower_topn)
                has_njobs = 'n_jobs' in signature.parameters
                if has_njobs:
                    ids_c = gower.gower_topn(np.array(representative_sample).reshape(1, -1), X_sample, n=nn.n_neighbors,
                                             cat_features=categorical, n_jobs=n_jobs)['index']
                else:
                    ids_c = gower.gower_topn(np.array(representative_sample).reshape(1, -1), X_sample, n=nn.n_neighbors,
                                             cat_features=categorical)['index']
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
        """ Predicts the outcome with an explainable model previously fitted

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

        return [int(f['name']) for f in self.uid3.predict(X)]

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

        if to_dict:
            return [self.uid3.tree.justification_tree(i, self.attributes_names).to_dict(reduce=reduce) for i in X.to_numpy()]
        else:
            return [self.uid3.tree.justification_tree(i, self.attributes_names).to_pseudocode(reduce=reduce) for i in X.to_numpy()]

    def __get_covered(self, rule, dataset, features, categorical=None):
        """ Returns covered instances from a given dataset and a rule

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
            op = ''  # if dict(zip(features, categorical))[i] == False else '=='
            query.append(f'{i}{op}' + f' and {i}{op}'.join(v))
        
        pattern = r'\b[A-Za-z]+(?:[^a-zA-Z\s0-9]+[A-Za-z]+)*(?:\s[A-Za-z]+(?:[^a-zA-Z\s0-9]+[A-Za-z]+)*)*\b'
        query = ' and '.join(query)
        all_names = set(re.findall(pattern, query.replace("and", "")))
        
        for name in all_names:
            new = '`' + name + '`'
            query = query.replace(name, new)

        covered = dataset.query(query)
        return covered

    def counterfactual(self, instance_to_explain, background, counterfactual_representative='medoid', reduce=True,
                       topn=None, n_jobs=None):
        """ Generates a counterfactual for a given instance and background data

        :param instance_to_explain:
        :param background:
        :param counterfactual_representative:
        :param reduce:
        :param topn:
        :param n_jobs:
        :return:
        """
        not_class = np.argmax(self.predict_proba(self.process_input(instance_to_explain)))
        rules = self.uid3.tree.to_dict(reduce=reduce)
        bbox_predictions = np.argmax(self.predict_proba(self.process_input(background)), axis=1)
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
                        distances = gower.gower_matrix(rule['covered'])
                        ids = np.argmin(distances.sum(axis=0))
                        dist = sklearn.metrics.pairwise_distances(rule['covered'].iloc[ids].values.reshape(1, -1),
                                                                  instance_to_explain)
                        representative_sample = rule['covered'].iloc[ids]
                        rule['counterfactual'] = representative_sample
                        rule['distance'] = dist
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
                        signature = inspect.signature(gower.gower_topn)
                        has_njobs = 'n_jobs' in signature.parameters
                        if has_njobs:
                            ids_dist = gower.gower_topn(instance_to_explain, rule['covered'], n=1,
                                                        cat_features=self.categorical, n_jobs=n_jobs)
                        else:
                            ids_dist = gower.gower_topn(instance_to_explain, rule['covered'], n=1,
                                                        cat_features=self.categorical)
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

    def visualize(self, data, target_column_name='class', fmt='.2f', instance2explain=None, counterfactual=None,
                  filename='tree.dot'):
        if counterfactual is not None:
            cfdf = pd.DataFrame(counterfactual['counterfactual']).T
            cfdf[target_column_name] = np.argmax(self.predict_proba(self.process_input(cfdf.values.reshape(1, -1)))[0])
        else:
            cfdf = None
        if instance2explain is not None:
            i2edf = pd.DataFrame(instance2explain, columns=self.attributes_names)
            i2edf[target_column_name] = np.argmax(
                self.predict_proba(self.process_input(i2edf.values.reshape(1, -1)))[0])
        else:
            i2edf = None
        self.tree.save_dot(filename, fmt=fmt, visual=True, background_data=data, instance2explain=i2edf,
                                counterfactual=cfdf)
    
    def to_dot(self, filename='tree.dot', fmt='.2f'):
        self.tree.save_dot(filename, fmt)

    def to_HMR(self):
        """ Exports to HMR format that can be executed by the HeaRTDroid rule-engine

        :return:
        """
        return self.tree.to_HMR()