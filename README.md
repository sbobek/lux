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


