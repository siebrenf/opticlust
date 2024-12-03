# PyClustree

Single cell clustering at a glance. 
Identify which clustering resolution(s) fit your data within minutes.

Opticlust currently offers:
  - Automated clustering (leiden/louvain) at various resolutions
  - Automatic selection of significant resolutions
  - Visualization of clusters per resolution and their relative compositions 
  - Easy to use, yet highly customizable Python API
  - Cluster recoloring for PyClustree and UMAP visualization (see below)

Opticlust will offer:
  - Clustering recommendation based on intra- and intercluster metrics

# Installation

```sh
git clone https://github.com/siebrenf/opticlust.git
pip install opticlust
```

## Dev install

```sh
git clone https://github.com/siebrenf/opticlust.git
conda env create -n opticlust -f opticlust/requirements.yaml
conda activate opticlust
pip install --editable ./opticlust --no-deps --ignore-installed
```

# Tutorial output

![](imgs/cluster_plot.png)

![](imgs/tree_plot_recolored.png)

![](imgs/umaps_recolored.png)

![](imgs/top_genes.png)

The UMAPs and cluster tree can be compared immediately due to the automatic renaming and recoloring of the clusters. 
Without renaming and recoloring, figures would have looked like this:

For example, note how cluster 2 becomes cluster 3 at resolution 0.53.

![](imgs/tree_plot_default.png)

![](imgs/umaps_default.png)


# Acknowledgements

This tool was inspired by:
- The original [Clustree](https://github.com/lazappi/clustree) R package.
- This [BioStars post](https://www.biostars.org/p/9489313/#9489342) by firestar.
