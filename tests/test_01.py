"""
run from the command line after navigating inside the opticlust dir with:
    pytest --disable-pytest-warnings -vvv
"""

import subprocess as sp
from os.path import dirname, join

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from opticlust.clust import clustering, clustering_plot
from opticlust.tree import clustree, clustree_plot

matplotlib.use("agg")  # This stop images from showing and blocking pytest


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


def test_load_data_fail():
    with pytest.raises(Exception) as excinfo:
        adata = pd.DataFrame()
        isinstance(adata.obs, pd.DataFrame)
    assert str(excinfo.value) == "'DataFrame' object has no attribute 'obs'"


# Load in data if above tests work
adata = sc.datasets.pbmc68k_reduced()


def test_clustering_positive():
    columns = clustering(adata.copy())
    assert isinstance(columns, list) == 1
    assert len(columns) == 81


def test_clustering_fail():
    with pytest.raises(Exception) as excinfo:
        columns = clustering(anndata)
    assert str(excinfo.value) == "name 'anndata' is not defined"


# Define the columns if above tests work
columns = clustering(adata)


# Run unit tests on clustering plot and check length of list
def test_clusteringplot_positive():
    tree_columns = clustering_plot(adata.copy(), columns)
    assert isinstance(tree_columns, list) == 1
    assert len(tree_columns) == 11


def test_clusteringplot_fail():
    with pytest.raises(Exception) as excinfo:
        columns = ["random1", "random2"]
        tree_columns = clustering_plot(adata.copy(), columns)
    assert (
        str(excinfo.value) == "Column names must be in the shape '[method]_res_[res]'"
    )


tree_columns = clustering_plot(adata, columns)


# Run unit tests on graph building and check if graph contains all
def test_buildtree_tree_data():
    tree_data = clustree(adata.copy(), tree_columns, rename_cluster=True)
    assert isinstance(tree_data, dict) == 1
    assert "graph" in tree_data
    assert "dimensions" in tree_data
    assert "axis" in tree_data


# Run unit tests on graph building and check if graph contains all
def test_buildtree_tree_data_rename_false():
    tree_data = clustree(adata.copy(), tree_columns, rename_cluster=False)
    assert isinstance(tree_data, dict) == 1
    assert "graph" in tree_data
    assert "dimensions" in tree_data
    assert "axis" in tree_data


# Use column instead of tree_columns if users do not want tree columns
adata2 = sc.datasets.pbmc68k_reduced()
columns2 = clustering(adata2)  # , samples=3


def test_buildtree_column_data():
    tree_data2 = clustree(adata2.copy(), columns2, rename_cluster=True)
    assert isinstance(tree_data2, dict) == 1
    assert "graph" in tree_data2
    assert "dimensions" in tree_data2
    assert "axis" in tree_data2


def test_buildtree_column_data_rename_false():
    tree_data2 = clustree(adata2.copy(), columns2, rename_cluster=False)
    assert isinstance(tree_data2, dict) == 1
    assert "graph" in tree_data2
    assert "dimensions" in tree_data2
    assert "axis" in tree_data2


def test_buildtree_fail():
    with pytest.raises(Exception) as excinfo:
        tree_columns3 = ["random1", "random2"]
        tree_data3 = clustree(adata.copy(), tree_columns3, rename_cluster=True)
    assert str(excinfo.value) == "columns not found in adata.obs: 'random1'"


def test_plottree_tree_data():
    tree_data = clustree(adata.copy(), tree_columns, rename_cluster=False)
    clustree_plot(tree_data)


def test_plottree_tree_data_rename_cluster():
    tree_data = clustree(adata.copy(), tree_columns, rename_cluster=True)
    clustree_plot(tree_data)
