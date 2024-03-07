.. lux-explainer documentation master file, created by
   sphinx-quickstart on Fri Feb 23 10:20:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the LUX documentation
---------------------------------

.. image:: https://raw.githubusercontent.com/sbobek/lux/main/pix/lux-logo.png
    :width: 240px
    :align: center
    :alt: Explanation-Tree

**LUX (Local Universal Rule-Based Explainer)** is an XAI algorithm that produces explanations for any type of machine-learning model.
It provides local explanations in a form of human-readable (and executable) rules, but also provide counterfactual explanations as well as visualization of the explanations.

Install
=======

LUX can be installed from either `PyPI <https://pypi.org/project/lux-explainer>`_ or directly from source code `GitHub <https://github.com/sbobek/lux>`_

To install form PyPI::

   pip install lux-explainer

To install from source code::

   git clone https://github.com/sbobek/lux
   cd lux
   pip install .

.. toctree::
   :maxdepth: 2
   :caption: Examples

   Basic Usage examples <basic_examples>
   SHAP-guided explanation generation examples <shap_examples>


.. toctree::
   :maxdepth: 2
   :caption: Reference

   API reference <api>

.. toctree::
   :maxdepth: 1
   :caption: Development

   Release notes <release_notes>
   Contributing guide <contributing>


