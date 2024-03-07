SHAP-guided explanations
=============

Adding SHAP to the explainer will make LUX try to build the explanation tree that is consistent with SHAP values.
In some cases this is important feature, especially when the model is analysed not only with LUX, but also with SHAP.

Yt also helps reducing so called Rashomon effect, because the LUX model uses the same features as blackbox model and therefore the explanations are suppose to be compliant with what really is happening in the balckbox model.

The full example with multiple datasets can be found here: `Notebook <https://github.com/sbobek/lux/blob/main/examples/lux_usage_example_shap.ipynb>`_

