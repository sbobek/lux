Basic usage examples
=============

The basic usage of LUX is similar to usage oof any of the scikit models that contains fit and predict functions.

.. code-block:: python

    from lux.lux import LUX
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    import numpy as np
    import pandas as pd

    # import some data to play with
    iris = datasets.load_iris()
    features = ['sepal_length','sepal_width','petal_length','petal_width']
    target = 'class'

    #create daatframe with columns names as strings (LUX accepts only DataFrames with string columns names)
    df_iris = pd.DataFrame(iris.data,columns=features)
    df_iris[target] = iris.target

    #train classifier
    train, test = train_test_split(df_iris)
    clf = svm.SVC(probability=True)
    clf.fit(train[features],train[target])
    clf.score(test[features],test[target])

    #pick some instance from dataset
    iris_instance = train[features].sample(1).values

    #train lux on neighbourhood equal 20 instances
    lux = LUX(predict_proba = clf.predict_proba,
        neighborhood_size=20,max_depth=2,
        node_size_limit = 1,
        grow_confidence_threshold = 0 )
    lux.fit(train[features], train[target], instance_to_explain=iris_instance,class_names=[0,1,2])

    #see the justification of the instance being classified for a given class
    lux.justify(np.array(iris_instance))
