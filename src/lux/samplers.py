# add smote and importance sampler__all__ = ['UncertainSMOTE']

from imblearn.over_sampling._smote.base import BaseSMOTE
import numpy as np
import warnings
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.metrics import pairwise_distances
import shap
import pandas as pd

from sklearn.linear_model import LinearRegression
import numdifftools as nd


class ImportanceSampler(TransformerMixin, BaseEstimator):
    def __init__(self, classifier, predict_proba, indstance2explain,min_generate_samples):
        self.classifier = classifier
        self.predict_proba = predict_proba
        self.instance2explain = indstance2explain
        self.min_generate_samples=min_generate_samples

    def fit(self, X,y=None):
        self.shap_values =  self.__getshap(X)
        return self

    def transform(self,X,y=None):
        return self.__importance_sampler(X,self.instance_to_explain, num=10)

    def __getshap(self, X_train_sample):
        """ Calculates SHAP values

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
        """ Generates data based on shapley values to minimize number of artificial samples.
        It generates samples only in the direction pointed by the gradient of SHAP

        :param X_train_sample:
        :param instance_to_explain:
        :param num:
        :return:
        """
        shap_values = self.shap_values
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
        fulldf_all = pd.concat([X_train_sample.reset_index(drop=True), shapdf.reset_index(drop=True)], axis=1)
        fulldf_all.index = X_train_sample.index
        class_of_i2e = self.classifier.predict(instance_to_explain.reshape(1, -1))
        predictions = self.classifier.predict(fulldf_all[cols])
        fulldf = fulldf_all[predictions == class_of_i2e]
        if len(fulldf) == 0:
            fulldf = fulldf_all[predictions != class_of_i2e]

        gradsf = {}

        for cl in np.unique(indexer):
            gradcl = []
            for dim in range(0, X_train_sample.shape[1]):
                mask = indexer == cl
                xs = X_train_sample.iloc[mask, dim]
                ys = shapclass[mask, dim]
                svr = LinearRegression()  # SVR()
                svr.fit(xs.values.reshape(-1, 1), ys)

                F = lambda x, svr=svr: svr.predict(x.reshape(1, -1))
                gradient = nd.Gradient(F)

                gradcl.append(gradient)
            gradsf[cl] = gradcl


        alpha = abs(shapclass).mean()

        def perturb(x, num, alpha, gradients, cols, shapcols):
            newx = []
            last = x[cols].values
            newx.append(last)
            cl = self.classifier.predict(last.reshape(1, -1))[0]

            grad = np.array([g(last[i]) for i, g in enumerate(gradients[cl])])
            for _ in range(0, num):
                last = last - alpha * grad
                cl = self.classifier.predict(last.reshape(1, -1))[0]
                grad = np.array([g(last[i]) for i, g in enumerate(gradients[cl])])
                newx.append(last)
                if cl != self.classifier.predict(last.reshape(1, -1))[0]:
                    break
            return np.array(newx)

        if fulldf.shape[0] > 0:
            upsamples = np.concatenate(
                fulldf.sample(int(self.min_generate_samples * len(fulldf))).apply(perturb, args=(
                    num, alpha, gradsf, cols),
                                                                                  axis=1).values)
            upsamples = upsamples[
                        np.random.choice(upsamples.shape[0], max(len(fulldf), upsamples.shape[0]), replace=False), :]
        else:
            upsamples = fulldf

        return pd.concat((pd.DataFrame(upsamples, columns=X_train_sample.columns), X_train_sample))


class UncertainSMOTE(BaseSMOTE):

    def __init__(
            self,
            *,
            predict_proba,
            sampling_strategy="all",
            random_state=None,
            k_neighbors=5,
            n_jobs=None,
            sigma=1,
            m_neighbors=10,
            min_samples=0.1,
            instance_to_explain=None,
            kind="borderline-1",
    ):
        """

        :param predict_proba:
        :param sampling_strategy:
        :param random_state:
        :param k_neighbors:
        :param n_jobs:
        :param sigma:
        :param m_neighbors:
        :param min_samples:
        :param instance_to_explain:
        :param kind:
        """
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.m_neighbors = m_neighbors
        self.kind = kind
        self.predict_proba = predict_proba
        self.sigma = sigma
        self.min_samples = min_samples
        self.instance_to_explain = instance_to_explain

    def _fit_resample(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        # FIXME: to be removed in 0.12
        if self.n_jobs is not None:
            warnings.warn(
                "The parameter `n_jobs` has been deprecated in 0.10 and will be "
                "removed in 0.12. You can pass an nearest neighbors estimator where "
                "`n_jobs` is already set instead.",
                FutureWarning,
            )

        self._validate_estimator()

        X_resampled = X.copy()
        y_resampled = y.copy()

        _, counts = np.unique(y_resampled, return_counts=True)
        samples_max = max(counts)
        additional_samples = int(self.min_samples * samples_max)

        # n_samples can be passed as an argument, n_iterations can also be used here
        for class_sample, n_samples in self.sampling_strategy_.items():
            n_samples += additional_samples

            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            # self.nn_m_.fit(X)
            danger_index = self._in_danger_noise(
                # self.nn_m_,
                self.predict_proba,
                X_class, class_sample, y, kind="danger"
            )
            if not any(danger_index):
                continue

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(
                _safe_indexing(X_class, danger_index), return_distance=False
            )[:, 1:]

            # divergence between borderline-1 and borderline-2
            if self.kind == "borderline-1":
                # Create synthetic samples for borderline points.
                X_new, y_new = self._make_samples(
                    _safe_indexing(X_class, danger_index),
                    y.dtype,
                    class_sample,
                    X_class,
                    nns,
                    n_samples,
                )
                if sparse.issparse(X_new):
                    X_resampled = sparse.vstack([X_resampled, X_new])
                else:
                    X_resampled = np.vstack((X_resampled, X_new))
                y_resampled = np.hstack((y_resampled, y_new))

            elif self.kind == "borderline-2":
                random_state = check_random_state(self.random_state)
                fractions = random_state.beta(10, 10)

                # only minority
                X_new_1, y_new_1 = self._make_samples(
                    _safe_indexing(X_class, danger_index),
                    y.dtype,
                    class_sample,
                    X_class,
                    nns,
                    int(fractions * (n_samples + 1)),
                    step_size=1.0,
                )

                # we use a one-vs-rest policy to handle the multiclass in which
                # new samples will be created considering not only the majority
                # class but all over classes.
                X_new_2, y_new_2 = self._make_samples(
                    _safe_indexing(X_class, danger_index),
                    y.dtype,
                    class_sample,
                    _safe_indexing(X, np.flatnonzero(y != class_sample)),
                    nns,
                    int((1 - fractions) * n_samples),
                    step_size=0.5,
                )

                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack([X_resampled, X_new_1, X_new_2])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1, X_new_2))
                y_resampled = np.hstack((y_resampled, y_new_1, y_new_2))

        return X_resampled, y_resampled

    def _in_danger_noise(self, predict_proba, samples, target_class, y, kind="danger"):
        """
        Estimate if a set of sample are in danger or noise.
        :param predict_proba:
           function returning probability estimates for samples
        :param samples: {array-like, sparse matrix} of shape (n_samples, n_features).
           The samples to check if either they are in danger or not.
        :param target_class: int or str.
           The target corresponding class being over-sampled.
        :param y: array-like of shape (n_samples,).
           The true label in order to check the neighbour labels.
        :param kind:
           {'danger', 'noise'}, default='danger'
            The type of classification to use. Can be either:
            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        :return: ndarray of shape (n_samples,).
           A boolean array where True refer to samples in danger or noise.
        """

        c_labels = samples[np.argmax(self.predict_proba(samples), axis=1) == target_class]
        prediction_certainty = np.max(self.predict_proba(c_labels), axis=1)

        # shuld this be thresholded like that, or keep n-lowest?
        confidence_threshold = np.mean(prediction_certainty) - self.sigma * np.std(
            prediction_certainty)  # changed to + (plus)

        if self.instance_to_explain is not None:
            distances = pairwise_distances(self.instance_to_explain.reshape(1, -1), c_labels)
            distancee_threshold = np.mean(distances) - self.sigma * np.std(distances)
            distance_mask = (distances < distancee_threshold)[0]
        else:
            distance_mask = (prediction_certainty < 0)

        if kind == "danger":
            return np.bitwise_or(prediction_certainty < confidence_threshold,  # changed fom <
                                 distance_mask)
        else:
            return prediction_certainty < 0  # always false
