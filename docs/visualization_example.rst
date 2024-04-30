Visualization examples
=============

Lux offers several ways of visualizing explanations:

* Textual (not considered visualization in fact),
* Basic Visual Graphviz-based visualization
* Advanced visual decision tree with data distributions, and factual and counterfactual paths
* Decision table visualization with `HeaRTDroid <https://heartdroid.re/>`_ rule obtained from `HWed <https://heartdroid.re/hwed/#/>`_ editor.


Textual
-----------
The basic explanation visualization is textual rule.
Calling `lux.justify` after fitting the model allows to generate rule-based explanation.

.. code-block:: python

    #see the justification of the instance being classified for a given class
    lux.justify(np.array(iris_instance))

    # This will generate something like:
    # The number after the hash mark indicate the confidence of the rule
    #['IF petal_length >=4.8999998569488525 THEN class = 2 # 1.0\n']

Graphviz
-----------

LUX allows to export justification tree to Graphviz format.

Calling following code after fitting the LUX model will export the justification tree to `tree.dot` and in Jupyter Notebook it will call `dot` to create PNG file and show it in notebook cell:

.. code-block:: python

    lux.uid3.tree.save_dot('tree-wine-shap.dot',fmt='.2f')
    gvz=graphviz.Source.from_file('tree.dot')
    !dot -Tpng tree-wine-shap.dot > tree-wine-shap.png
    Image('tree.png')

The result of that call would be something like:

.. image:: https://raw.githubusercontent.com/sbobek/lux/main/pix/tree-gv.png
    :alt: Explanation-Tree

Advanced
-----------
The more advanced, but also the easiest way to obtain visualizations is by calling `vizualize` function of LUX.
Additionally passing instance2explain and counterfactual parameters will allow to mark their decision path on the tree with red and blue dots.

.. code-block:: python

    lux.visualize(data=train,target_column_name='class',instance2explain=i2e,counterfactual=cf,filename='tree-wine.dot' )
    gvz=graphviz.Source.from_file('tree-wine-shap.dot')
    !dot -Tpng tree-wine-shap.dot > tree-wine-shap.png
    Image('tree-wine-shap.png')


The result will look like tat:

.. image:: https://raw.githubusercontent.com/sbobek/lux/main/pix/tree-wine-shap.png
    :alt: Explanation-Tree

Decision Table
-----------

The last way of visualizing LUX decision is via decision table.
LUX allows to export decision tree to HRM format which is used by the Heartdroid software, and by HWEd editor.

.. code-block:: python

    print(lux.to_HMR())

The output can be imported into HWED editor here: `HWED Online <https://heartdroid.re/hwed/>`_ and the generated output will be looked like that


.. image:: https://raw.githubusercontent.com/sbobek/lux/main/pix/xtt-table.png
    :alt: Decision-table