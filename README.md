# Source code for benchmarks of LUX, LORE, Anchor and EXPLAN
##Structure

1. `lux_benchmark-unbiased-openml-ablation` - the main file with benchmark combined with ablation studies
2. `lux_paper_evaluation_figures` - notebook for figures used in the paper describing LUX
3. `lux_paper_evaluation_figures-ablation` notebook for figures used in the paper describing LUX (ablation study)
4. `lux_paper_figures_visualizations` - visualizations of decision boundaries, sampling mechanisms, etc. used in the paper

## Requirements
The benchmark requires extended packages compared to pure LUX package, hence, in order to run the code, follow:

```
conda create --name luxenv python=3.8
conda activate luxenv
conda install pip
pip install -r benchmark-requirements.txt
```

## Evaluaiton results data
Following folders contains evalaution studies results:

| Directory name                                     | Classifier    | Dataset source                  | Oblique splits | SHAP-based importance | Description                                                                                                                                 |
| -------------------------------------------------- | ------------- | ------------------------------- | -------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `ablation-full-knn-nosmote+oblique+shap`           | kNN           | OpenML Benchmark Suite          | ✅ Enabled      | ✅ Enabled             | Ablation study using kNN as black-box, evaluated on OpenML Benchmark Suite datasets, with oblique splits and SHAP-based feature importance. |
| `ablation-full-knn-nosmote-nooblique+shap`         | kNN           | OpenML Benchmark Suite          | ❌ Disabled     | ✅ Enabled             | Same as above, but restricted to axis-aligned (non-oblique) splits.                                                                         |
| `ablation-full-knn-nosuite-nosmote+oblique+shap`   | kNN           | OpenML (general classification) | ✅ Enabled      | ✅ Enabled             | kNN-based ablation evaluated on general OpenML classification datasets, with oblique splits and SHAP importance.                            |
| `ablation-full-knn-nosuite-nosmote-nooblique+shap` | kNN           | OpenML (general classification) | ❌ Disabled     | ✅ Enabled             | kNN ablation on general OpenML datasets using only axis-aligned splits and SHAP importance.                                                 |
| `ablation-full-mlp-nosmote+oblique+shap`           | MLP           | OpenML Benchmark Suite          | ✅ Enabled      | ✅ Enabled             | Ablation study using an MLP black-box model, evaluated on OpenML Benchmark Suite datasets, with oblique splits and SHAP importance.         |
| `ablation-full-mlp-nosmote-nooblique+shap`         | MLP           | OpenML Benchmark Suite          | ❌ Disabled     | ✅ Enabled             | Same MLP-based evaluation as above, but without oblique splits.                                                                             |
| `ablation-full-mlp-nosuite-nosmote+oblique+shap`   | MLP           | OpenML (general classification) | ✅ Enabled      | ✅ Enabled             | MLP ablation evaluated on general OpenML datasets, with oblique splits and SHAP-based importance.                                           |
| `ablation-full-mlp-nosuite-nosmote-nooblique+shap` | MLP           | OpenML (general classification) | ❌ Disabled     | ✅ Enabled             | MLP ablation on general OpenML datasets, restricted to axis-aligned splits, using SHAP importance.                                          |
| `ablation-full-rfc-nosmote+oblique+shap`           | Random Forest | OpenML Benchmark Suite          | ✅ Enabled      | ✅ Enabled             | Ablation study using Random Forest as black-box, evaluated on OpenML Benchmark Suite datasets, with oblique splits and SHAP importance.     |
| `ablation-full-rfc-nosmote-nooblique+shap`         | Random Forest | OpenML Benchmark Suite          | ❌ Disabled     | ✅ Enabled             | Random Forest ablation on benchmark datasets without oblique splits, using SHAP importance.                                                 |
| `ablation-full-rfc-nosuite-nosmote+oblique+shap`   | Random Forest | OpenML (general classification) | ✅ Enabled      | ✅ Enabled             | Random Forest ablation evaluated on general OpenML datasets, with oblique splits and SHAP-based feature importance.                         |
| `ablation-full-rfc-nosuite-nosmote-nooblique+shap` | Random Forest | OpenML (general classification) | ❌ Disabled     | ✅ Enabled             | Random Forest ablation on general OpenML datasets using axis-aligned splits and SHAP importance.                                            |

Each folder contains set to CSV files with all the results from evaluation:

| File name                       | Metric / Content        | Description                                                                                                                                                                                   |
| ------------------------------- |-------------------------| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `synthx100_scores.csv`          | Counterfactual fidelity | Measures **fidelity of the explanation under counterfactual perturbations**, i.e. how well the rule-based explanation preserves the black-box model’s predictions when features are modified. |
| `synthx100_importance_gain.csv` | SHAP consistency        | Quantifies **alignment between explanation rules and SHAP feature importance**, measuring how much of the total SHAP importance mass is covered by features used in the explanation.          |
| `synthx100_hits.csv`            | Hit rate                | Binary indicator of whether the **explanation rule predicts the same class** as the black-box model for the explained instance.                                                               |
| `synthx100_times.csv`           | Runtime                 | Time (in seconds) required to **generate a local explanation** for a single instance, including sampling and tree induction.                                                                  |
| `synthx100_confidences.csv`     | Explanation confidence  | Confidence of the generated explanation, typically derived from **rule coverage and class probability** within the induced local model.                                                       |
| `synthx100_stability.csv`       | Stability (real data)   | Measures **stability of explanations under small perturbations of real data**, i.e. how consistent feature selections and rules remain across similar instances.                              |
| `synthx100_nac.csv`             | NAC (local accuracy)    | **Nearest-neighbour accuracy**: accuracy of the explanation model on the neighbourhood of the explained instance (local region fidelity).                                                     |
| `synthx100_rulecov.csv`         | Rule coverage           | Fraction of sampled instances that **satisfy the explanation rule**, reflecting how general the rule is in the local neighbourhood.                                                           |
| `synthx100_rulecov_nn.csv`      | Rule coverage (NN)      | Rule coverage computed **only on nearest neighbours** of the explained instance, emphasizing strictly local generality.                                                                       |
| `synthx100_local_fid.csv`       | Local fidelity          | Fidelity of the explanation model **with respect to the black-box predictions** on locally generated samples.                                                                                 |
| `synthx100_local_fid_nn.csv`    | Local fidelity (NN)     | Local fidelity measured **only on nearest neighbours**, analogous to `*_nn` NAC-style metrics.                                                                                                |
| `synthx100_rulelen.csv`         | Explanation length      | Number of conditions (features) in the extracted explanation rule, serving as a **complexity / interpretability proxy**.                                                                      |
