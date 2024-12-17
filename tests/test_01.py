"""
run from the command line (from the opticlust directory) with:
    pytest --disable-pytest-warnings -vvv
"""

import os
import subprocess as sp
from os.path import dirname, join

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from opticlust.clust import clustering, clustering_plot
from opticlust.recommend import resolutionrecommender
from opticlust.tree import clustree, clustree_plot

matplotlib.use("agg")  # This stop images from showing and blocking pytest
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too late to lint!")
def test_black_lint():
    base = dirname(dirname(__file__))
    sp.check_output(
        "black " + f"{join(base, 'opticlust')} {join(base, 'tests')}",
        shell=True,
    )
    sp.check_output(
        "isort --overwrite-in-place --profile black --conda-env requirements.yaml "
        + f"{join(base, 'opticlust')} {join(base, 'tests')}",
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


def test_resolutionrecommender_tree_data_median():
    all, low, medium, high = resolutionrecommender(adata.copy(), tree_columns)
    assert all == "leiden_res_0.03"
    assert low == "leiden_res_0.03"
    assert medium == "leiden_res_0.88"
    assert high == "leiden_res_1.60"


def test_resolutionrecommender_column_data_median():
    all, low, medium, high = resolutionrecommender(
        adata2.copy(), columns2, rank_method="median", tests="SH_CH_DB"
    )
    assert all == "leiden_res_0.03"
    assert low == "leiden_res_0.03"
    assert medium == "leiden_res_1.07"
    assert high == "leiden_res_1.43"


#
def test_resolutionrecommender_column_data_median_order():
    all, low, medium, high = resolutionrecommender(
        adata2.copy(), columns2, rank_method="mean", tests="DB_CH_SH"
    )
    assert all == "leiden_res_1.18"
    assert low == "leiden_res_0.03"
    assert medium == "leiden_res_1.18"
    assert high == "leiden_res_1.30"


def test_resolutionrecommender_column_data_orderSH():
    all, low, medium, high = resolutionrecommender(
        adata2.copy(), columns2, rank_method="order", tests="SH_CH_DB"
    )
    assert all == "leiden_res_0.03"
    assert low == "leiden_res_0.03"
    assert medium == "leiden_res_1.07"
    assert high == "leiden_res_1.30"


def test_resolutionrecommender_column_data_orderCH():
    all, low, medium, high = resolutionrecommender(
        adata2.copy(), columns2, rank_method="order", tests="CH_DB_SH"
    )
    assert all == "leiden_res_1.38"
    assert low == "leiden_res_0.68"
    assert medium == "leiden_res_1.20"
    assert high == "leiden_res_1.38"


def test_resolutionrecommender_column_data_orderDB():
    all, low, medium, high = resolutionrecommender(
        adata2.copy(), columns2, rank_method="order", tests="DB_CH_SH"
    )
    assert all == "leiden_res_0.03"
    assert low == "leiden_res_0.03"
    assert medium == "leiden_res_1.05"
    assert high == "leiden_res_1.43"


def test_resolutionrecommender_column_data_modeFail():
    with pytest.raises(Exception) as excinfo:
        resolutionrecommender(
            adata2.copy(), columns2, rank_method="mode", tests="DB_CH_SH"
        )
    assert str(excinfo.value) == "rank_method must be: median, mean or order"
