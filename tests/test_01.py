"""
run from the command line after navigating inside the opticlust dir with:
    pytest --disable-pytest-warnings -vvv
"""

import subprocess as sp
import warnings
from os.path import dirname, join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from opticlust.clust import clustering, clustering_plot
from opticlust.tree import clustree, clustree_plot

matplotlib.use("agg")  # This does not default plot images and lets pytest run


def test_black_lint():
    base = dirname(dirname(__file__))
    sp.check_output(
        "black " + f"{join(base, 'tests')}",
        shell=True,
    )
    sp.check_output(
        "isort --overwrite-in-place --profile black --conda-env requirements.yaml "
        + f"{join(base, 'tests')}",
        shell=True,
    )


def test_load_data_positive():
    # load data (adata.obs should be a pd.DataFrame)
    adata = sc.datasets.pbmc68k_reduced()
    assert isinstance(adata.obs, pd.DataFrame) == 1
    assert isinstance(adata.var, pd.DataFrame) == 1
    assert isinstance(adata.X, np.ndarray) == 1


def test_load_data_negative():
    # load data (adata.obs should not be anything else)
    adata = sc.datasets.pbmc68k_reduced()
    assert not isinstance(adata.obs, pd.DataFrame) == 0
    assert not isinstance(adata.var, pd.DataFrame) == 0
    assert not isinstance(adata.X, np.ndarray) == 0


# Load in data if above tests work
adata = sc.datasets.pbmc68k_reduced()


def test_clustering_positive():
    columns = clustering(adata)
    assert isinstance(columns, list) == 1
    assert len(columns) == 81


def test_clustering_negative():
    columns = clustering(adata)
    assert not isinstance(columns, list) == 0


# Define the columns if above tests work
columns = clustering(adata)


# Run unit tests on clustering plot and check length of list
def test_clusteringplot_positive():
    tree_columns = clustering_plot(adata, columns)
    plt.close()
    assert isinstance(tree_columns, list) == 1
    assert len(tree_columns) == 11


def test_clusteringplot_negative():
    tree_columns = clustering_plot(adata, columns)
    plt.close()
    assert not isinstance(tree_columns, list) == 0


# Define tree columns if the above tests work
tree_columns = clustering_plot(adata, columns)


# Run unit tests on graph building and check if graph contains all
def test_buildtree():
    tree_data = clustree(adata, tree_columns, rename_cluster=True)
    assert isinstance(tree_data, dict) == 1
    assert "graph" in tree_data
    assert "dimensions" in tree_data
    assert "axis" in tree_data


# To do: implement testing the clustree_plot in some way?
