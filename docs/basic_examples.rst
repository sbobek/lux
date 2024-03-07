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



As a result you obtain an explanation that says that if petal_length is grater than 4.89 than the class is 2.

.. code-block::

    ['IF petal_length >=4.8999998569488525 THEN class = 2 # 1.0\n']

You can obtain  counterfactual explanation for this examples as well:

.. code-block:: python

    ## Generate counterfactual
    cf = lux.counterfactual(np.array(iris_instance), train[features], counterfactual_representative='nearest', topn=1)[0]
    print(f"Counterfactual for {iris_instance} to change from class {lux.predict(np.array(iris_instance))[0]} to class {cf['prediction']}: \n{cf['counterfactual']}")

As a result you will get

.. code-block::

    Counterfactual for [[7.7 3.  6.1 2.3]] to change from class 2 to class 1:
    sepal_length    7.0
    sepal_width     3.2
    petal_length    4.7
    petal_width     1.4

Finally, the explanations can be visulized with the following code:

.. code-block:: python

    import graphviz
    from graphviz import Source
    from IPython.display import SVG, Image

    i2edf = pd.DataFrame(iris_instance, columns=features)
    i2edf[target] =clf.predict(i2edf.values.reshape(1,-1))[0]
    cfdf = pd.DataFrame(cf['counterfactual']).T
    cfdf[target] = clf.predict(cfdf.values.reshape(1,-1))[0]
    lux.uid3.tree.save_dot('tree.dot',fmt='.2f',visual=True, background_data=train, instance2explain=i2edf, counterfactual=cfdf)
    gvz=graphviz.Source.from_file('tree.dot')
    !dot -Tpng tree.dot > tree.png
    Image('tree.png')


And as a result you get the following tree. Note that red dot represents the instance that we were explaining and blue dot represents the counterfactual for this instance.

.. image:: https://raw.githubusercontent.com/sbobek/lux/main/pix/basic-example.png
    :alt: Explanation-Tree