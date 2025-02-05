Troubleshooting
=============

In the following sections we provided common problems with using LUX with custom datasets and models.

1. **Dataframe columns names must be strings**

LUX accepts only DataFrames with string columns names.
If you have a DataFrame with columns names as integers or floats, you need to convert them to strings.
Additionally, in the current version of the software, the names of the columns cannot contain any special characters and whitespaces except for underscore: `_`.
This limitation is caused by the linear, oblique split conditions, that may misinterpret feature-names with special characters as mathematical operations of the linear equation split.

You can easily ensure that the columns names are strings by converting them to strings, as shown below:

.. code-block:: python

    import re
    features = [re.sub(r'[^\w]', '_', s) for s in my_df.columns]
    my_df.columns = features


2. **Model must have predict_proba method**

LUX requires a model to have a `predict_proba` method.
If you are using a model that does not have this method, you need to wrap it with a model that has this method.
Similarly, if your model requires special input preprocessing before classification (e.g. it only works with OHE categorical features, but you want LUX to produce explanations with original parameters.

You can cope with this issues by wrapping the model around the wrapper that preprocesses the input data before classification and implements the `predict_proba` method.
The example of tat was given below.
You can also find full working example here `Google Colab example <https://colab.research.google.com/drive/1Yb-VGzsJupTYyyuwA9dEVLYkftuyk4C8?usp=sharing>`_

.. code-block:: python

    from sklearn.base import BaseEstimator
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    class CategoricalWrapper(BaseEstimator):
        def __init__(self, model, ohe_encoder = None, categorical_indicator=None, features = None, categories='auto'):
            if ohe_encoder is None:
                self.ohe_encoder = OneHotEncoder(categories=categories)
            else:
                self.ohe_encoder = ohe_encoder

            self.features = features
            self.categories=categories
            self.categorical_indicator=categorical_indicator
            self.ct = ColumnTransformer(
                [("categorical", self.ohe_encoder, [f for f,c in zip(features,categorical_indicator) if c ] )],
            remainder='passthrough')

            self.model = model

        def fit(self, X,y):
            X_tr = self.ct.fit_transform(X)
            self.model.fit(X_tr,y)
            return self
        def predict(self,X):
            if type(X) is np.ndarray and self.features is not None:
                X = pd.DataFrame(X,columns=features)
            return self.model.predict(self.ct.transform(X))

        def predict_proba(self, X):
            if type(X) is np.ndarray and self.features is not None:
                X = pd.DataFrame(X,columns=features)
            return self.model.predict_proba(self.ct.transform(X))

What if your model does not have predict proba? If your model predicts only labels, without any way of transforming it into probability-like values,
you can always transform the probabilities into 0/1 values indicating 100% probability of one of the class, or use decision function to transform the output into probabilities.
The example of that is given below.

.. code-block:: python

    class ClassifierWrapper(BaseEstimator):
        def __init__(self, model):
            self.model = model

        def fit(self, X,y):
            self.model.fit(X,y)
            return self
        def predict(self,X):
            return self.model.predict(X)

        def predict_proba(self, X):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            elif hasattr(self.model, 'decision_function'):
                # Sigmoid transformation for decision_function output
                decision_scores = self.model.decision_function(X)
                probabilities = 1 / (1 + np.exp(-decision_scores))
                return np.column_stack([1 - probabilities, probabilities])
            else:
                return np.array([self.model.predict(X)==c for c in self.model.classes_]).T

3. **When calling visualize, the instance2explain and counterfactual parameters must be passed as DataFrames**

When calling visualize, the instance2explain and counterfactual parameters must be passed as DataFrames, even though they are single instances.
This is the limitation in the current version, we plan to remove it in the future.

.. code-block:: python

    import graphviz
    from graphviz import Source
    from IPython.display import SVG, Image

    #make it a DataFrame with one row
    i2edf = pd.DataFrame(iris_instance, columns=features)
    #make it a DataFrame with one row
    i2edf[target] =clf.predict(i2edf.values.reshape(1,-1))[0]
    cfdf = pd.DataFrame(cf['counterfactual']).T
    cfdf[target] = clf.predict(cfdf.values.reshape(1,-1))[0]
    lux.uid3.tree.save_dot('tree.dot',fmt='.2f',visual=True, background_data=train, instance2explain=i2edf, counterfactual=cfdf)
    gvz=graphviz.Source.from_file('tree.dot')
    !dot -Tpng tree.dot > tree.png
    Image('tree.png')

4. **LUX is designed only for classification problems**

In case you would like to use LUX for other ML tasks, such as regression, or anomaly detection, you need to wrap your model with a classifier that will provide probabilities of the class.
This is the same issue as in the case of `predict_proba` method, but in this case, you need to wrap the model with a classifier that output will be discrete (eg. discretized output variable for regression, binarized anomaly score with threshold indicating normal/abnormal class.


