# opticlust

[![CI/CD](https://github.com/siebrenf/opticlust/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/siebrenf/opticlust/actions/actions/workflows/ci-cd.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/opticlust.svg)](https://badge.fury.io/py/opticlust)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/opticlust/README.html)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/opticlust/badges/version.svg)](https://anaconda.org/bioconda/opticlust)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/opticlust/badges/downloads.svg)](https://anaconda.org/bioconda/opticlust)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14513541.svg)](https://doi.org/10.5281/zenodo.14513541)

Single cell clustering and recommendations at a glance. 
Identify which clustering resolution(s) fit your data within minutes.

Opticlust currently offers:
  - Automated clustering (leiden/louvain) at various resolutions
  - Automatic selection of significant resolutions
  - Clustering recommendations based on intra- and intercluster metrics
  - Visualization of clusters per resolution and their relative compositions 
  - Easy to use, yet highly customizable Python API
  - Cluster recoloring for opticlust and UMAP visualization (see below)


# Installation

## PyPi

```sh
pip install opticlust
```

## Conda

```sh
conda install -c bioconda opticlust
```

## GitHub

```sh
git clone https://github.com/siebrenf/opticlust.git
pip install opticlust
```

## Develop

```sh
git clone https://github.com/siebrenf/opticlust.git
conda env create -n opticlust -f opticlust/requirements.yaml
conda activate opticlust
pip install --editable ./opticlust --no-deps --ignore-installed
```


# Tutorial output

Output of `clustering_plot()` and `score_resolutions()`:
![clustering_plot](imgs/clustering_plot.png)
![score_resolutions](imgs/score_resolutions.png)

Output of `clustree_plot()`:
![clustree_plot_recolored](imgs/clustree_plot_recolored.png)

Output of `sc.pl.umap()`
![umaps_recolored](imgs/umaps_recolored.png)

Output of `sc.pl.rank_genes_groups_heatmap()` and `sc.pl.rank_genes_groups_dotplot()`:

`top_low` recommended resolution:
![top_genes_heatmap_0](imgs/top_genes_heatmap_0.png)
![top_genes_dotplot_0](imgs/top_genes_dotplot_0.png)

`top_medium` recommended resolution:
![top_genes_heatmap_1](imgs/top_genes_heatmap_1.png)
![top_genes_dotplot_1](imgs/top_genes_dotplot_1.png)

`top_high` recommended resolution:
![top_genes_heatmap_2](imgs/top_genes_heatmap_2.png)
![top_genes_dotplot_2](imgs/top_genes_dotplot_2.png)


# Advantages of opticlust

The UMAPs and cluster tree plot can be compared immediately due to the automatic renaming and recoloring of the clusters. 
Without renaming and recoloring, figures would have looked like this:

Output of `clustree(rename_cluster=False)` and `sc.pl.umap()`:
![clustree_plot_default](imgs/clustree_plot_default.png)
![umaps_default](imgs/umaps_default.png)
Note how cluster 2 becomes cluster 3 at resolution 0.50.
This makes it difficult to track how changes in resolution impacted the clustering.


# Acknowledgements

This tool was inspired by:
- The original [Clustree](https://github.com/lazappi/clustree) R package.
- This [BioStars post](https://www.biostars.org/p/9489313/#9489342) by firestar.

# How to cite
When using this software package, please correctly cite the accompanied DOI under "Citation": https://doi.org/10.5281/zenodo.14513541
