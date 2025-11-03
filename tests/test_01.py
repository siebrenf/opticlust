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
from opticlust.recommend import recommend_resolutions, score_resolutions
from opticlust.tree import clustree
from opticlust.utils import validate_resolutions

matplotlib.use("agg")  # This stop images from showing and blocking pytest
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture
def adata_clean():
    return sc.datasets.pbmc68k_reduced()


@pytest.fixture
def columns(adata_clean):
    return clustering(adata_clean)


@pytest.fixture
def adata_scored(adata_clean, columns):
    adata = adata_clean.copy()
    score_resolutions(adata, columns)
    return adata


@pytest.fixture
def tree_columns(adata_scored, columns):
    return clustering_plot(adata_scored, columns, method="score")


@pytest.fixture
def tree_data(adata_clean, tree_columns):
    return clustree(adata_scored, tree_columns, rename_cluster=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too late to lint!")
def test_black_lint():
    base = dirname(dirname(__file__))
    sp.check_output(
        "black "
        + f"{join(base, 'opticlust')} {join(base, 'tests')} {join(base, 'tutorial.py')}",
        shell=True,
    )
    sp.check_output(
        "isort --overwrite-in-place --profile black --conda-env requirements.yaml "
        + f"{join(base, 'opticlust')} {join(base, 'tests')} {join(base, 'tutorial.py')}",
        shell=True,
    )

    sp.check_output(
        "ruff check --line-length 88 --extend-select C4,SIM,TCH,E4,E7,E9,F --ignore E402 "
        + f"{join(base, 'opticlust')} {join(base, 'tests')} {join(base, 'tutorial.py')}",
        shell=True,
    )


def test_load_data_positive(adata_clean):
    assert isinstance(adata_clean.obs, pd.DataFrame)
    assert isinstance(adata_clean.var, pd.DataFrame)
    assert isinstance(adata_clean.X, np.ndarray)


def test_validate_resolutions():
    columns = ["leiden_res_0.00"]
    method_clustering, resolutions = validate_resolutions(columns)
    assert method_clustering == "leiden"
    assert resolutions == [0.00]

    columns = ["louvain_res_1.00", "louvain_res_2.00"]
    method_clustering, resolutions = validate_resolutions(columns)
    assert method_clustering == "louvain"
    assert resolutions == [1.00, 2.00]

    for columns in [
        ["Wrong_Number_Of_Underscores"],
        ["leiden_RandomText_0.00"],
        ["WrongMethod_res_0.00"],
        ["leiden_res_NotAnInt"],
    ]:
        with pytest.raises(Exception) as excinfo:
            validate_resolutions(columns)
        assert (
            str(excinfo.value)
            == "Column names must be in the shape '[method]_res_[res]'"
        )


def test_clustering_positive(adata_clean):
    columns = clustering(adata_clean, samples=81)
    assert isinstance(columns, list)
    assert len(columns) == 81


def test_score_resolutions_2_mean(adata_clean, columns):
    adata = adata_clean.copy()
    score_resolutions(adata, columns[0:5], tests="CH_SH", method="mean")
    assert "opticlust" not in adata_clean.uns
    assert "opticlust_params" not in adata_clean.uns
    assert "opticlust" in adata.uns
    assert "opticlust_params" in adata.uns
    assert adata.uns["opticlust_params"]["tests"] == "CH_SH"
    assert adata.uns["opticlust_params"]["method"] == "mean"


def test_score_resolutions_all_median(adata_clean, columns):
    adata = adata_clean.copy()
    score_resolutions(adata, columns[0:5], tests="SH_DB_CH", method="median")
    assert "opticlust" not in adata_clean.uns
    assert "opticlust_params" not in adata_clean.uns
    assert "opticlust" in adata.uns
    assert "opticlust_params" in adata.uns
    assert adata.uns["opticlust_params"]["tests"] == "SH_DB_CH"
    assert adata.uns["opticlust_params"]["method"] == "median"


def test_score_resolutions_2_order(adata_clean, columns):
    adata = adata_clean.copy()
    score_resolutions(adata, columns[0:5], tests="DB_SH", method="order")
    assert "opticlust" not in adata_clean.uns
    assert "opticlust_params" not in adata_clean.uns
    assert "opticlust" in adata.uns
    assert "opticlust_params" in adata.uns
    assert adata.uns["opticlust_params"]["tests"] == "DB_SH"
    assert adata.uns["opticlust_params"]["method"] == "order"


def test_score_resolutions_subset(adata_clean, columns):
    adata = adata_clean.copy()
    # no subset
    score_resolutions(adata, columns[-3:-1], max_n_silhouette=-1)
    # subset
    score_resolutions(adata, columns[-3:-1], max_n_silhouette=1)


def test_clusteringplot_middle(adata_clean, columns):
    tree_columns = clustering_plot(adata_clean, columns[0:5], method="middle")
    assert isinstance(tree_columns, list)
    assert len(tree_columns) == 2


def test_clusteringplot_score(adata_scored, columns):
    tree_columns = clustering_plot(adata_scored, columns[0:5], method="score")
    assert isinstance(tree_columns, list)
    assert len(tree_columns) == 2


def test_clusteringplot_median(adata_clean, columns):
    tree_columns = clustering_plot(adata_clean, columns[0:5], method="median")
    assert isinstance(tree_columns, list)
    assert len(tree_columns) == 2


def test_clusteringplot_fail(adata_clean, columns):
    with pytest.raises(Exception) as excinfo:
        clustering_plot(adata_clean, columns=["random1", "random2"])
    assert (
        str(excinfo.value) == "Column names must be in the shape '[method]_res_[res]'"
    )

    with pytest.raises(ValueError) as excinfo:
        clustering_plot(adata_clean, columns[0:5], method="score")
    assert str(excinfo.value) == "Please run score_resolutions() first!"


def test_recommendresolutions_tree_columns(adata_scored, tree_columns):
    overall, low, medium, high = recommend_resolutions(adata_scored, tree_columns)
    assert overall == "leiden_res_0.20"
    assert low == "leiden_res_0.20"
    assert medium == "leiden_res_1.07"
    assert high == "leiden_res_1.75"


def test_buildtree_fail(adata_clean):
    with pytest.raises(Exception) as excinfo:
        tree_columns3 = ["random1", "random2"]
        clustree(adata_clean, tree_columns3, rename_clusters=True)
    assert (
        str(excinfo.value) == "Column names must be in the shape '[method]_res_[res]'"
    )


def test_plottree_tree_data(adata_clean, tree_columns):
    clustree(adata_clean, tree_columns, rename_clusters=False)


def test_plottree_tree_data_rename_cluster(adata_clean, tree_columns):
    clustree(adata_clean, tree_columns, rename_clusters=True)
