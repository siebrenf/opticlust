import os
import subprocess as sp
from os.path import dirname, join

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from opticlust.clust import clustering, clustering_plot
from opticlust.tree import clustree, clustree_plot
#from opticlust.recommend import clusterrecommender

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

# code from opticlust.recommend
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import numpy as np
import warnings
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def resolutionrecommender(
    adata,
    columns,
    resolution_min = 0.0,
    resolution_max = 2.0,
    method = "median",
    test_order = "SH_DB_CH",
    figsize=(16, 8),
    subplot_kwargs=None,
    return_plot=False,
):
    """
    Recommends clustering resolutions based on scores of multiple clustering related tests.
    Silhoutte (SH), Davies Bouldin (DB) and Calinski-Harabasz (CH) scores are all calculated.

    :param adata: dataset
    :param columns: list of adata.obs column names to use in the plot
    :param resolution_min: the lowest clustering resolution
    :param resolution_max: the highest clustering resolution
    :param method: combines ranked scores from tests with the options "median", "mean" or "mode".
    If "order" is selected then test_order then ranking will be fully based on the test_order parameter.
    :param test_order: if "order" not chosen in method then ranks based on these values only if ties are found. 
    All possible combinations can be chosen, including: SH_CH_DB, DB_SH_CH etc. (default SH_DB_CH).
    :param figsize: matplotlib figsize
    :param subplot_kwargs: kwargs passed on to plt.subplot
    :param return_plot: if True, also returns fig and ax
    """

    if subplot_kwargs is None:
        subplot_kwargs = {}

    if columns[0].count("_") != 2:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    method = columns[0].split("_", 1)[0]
    if method not in ["leiden", "louvain"]:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")

    plotdf = sc.get.obs_df(
        adata,
        keys=[*columns],
	obsm_keys=[("X_umap", 0), ("X_umap", 1)]
    )
    
    dim1 = plotdf['X_umap-0'].to_numpy()
    dim2 = plotdf['X_umap-1'].to_numpy()
    dims = np.concatenate((dim1.reshape(-1,1),dim2.reshape(-1,1)),axis=1)

    sil_list=list()
    cal_list=list()
    dav_list=list()

    for i in columns:
        test_res = plotdf[i].to_numpy()
        try: 
            sil_list.append(silhouette_score(dims, test_res))
        except (ValueError, AttributeError) as e:
            sil_list.append(np.nan)
        try: 
            cal_list.append(calinski_harabasz_score(dims, test_res))
        except (ValueError, AttributeError) as e:
            cal_list.append(np.nan)
        try: 
            dav_list.append(davies_bouldin_score(dims, test_res))
        except (ValueError, AttributeError) as e:
            dav_list.append(np.nan)

    df_metrics=pd.DataFrame(list(zip(sil_list, cal_list, dav_list)),
            columns=["SH_score",
                     "CH_score",
                     "DB_score"])
    df_metrics["resolutions"]=columns
    df_metrics

    df = df_metrics

    # Normalize the scores with min-max scaling (0-1). DB is inverted because lower indicates better clustering.
    df['SH_score_normalized'] = (df['SH_score'] - 
                                         df['SH_score'].min()) / (df['SH_score'].max() - df['SH_score'].min())
    df['CH_score_normalized'] = (df['CH_score'] - 
                                                df['CH_score'].min()) / (df['CH_score'].max() - df['CH_score'].min())
    df['DB_score_normalized'] = 1 - (df['DB_score'] - 
                                                 df['DB_score'].min()) / (df['DB_score'].max() - df['DB_score'].min())

    # Rank the test scores
    first_score = test_order.split("_")[0]
    second_score = test_order.split("_")[1]
    third_score = test_order.split("_")[2]

    df[f'{first_score}_rank'] = df[f'{first_score}_score_normalized'
                                   ].rank(ascending=False)
    df[f'{second_score}_rank'] = df[f'{second_score}_score_normalized'
                                    ].rank(ascending=False)
    df[f'{third_score}_rank'] = df[f'{third_score}_score_normalized'
                                   ].rank(ascending=False)

    # Combine the ranks into a median single score (thinking of reducing repetitive code here)
    if method == "median":
        df['combined_rank'] = df[[f'{first_score}_rank', 
                                f'{second_score}_rank', 
                                f'{third_score}_rank']
                                ].median(axis=1)
    if method == "mean":
        df['combined_rank'] = df[[f'{first_score}_rank', 
                                f'{second_score}_rank', 
                                f'{third_score}_rank']
                                ].mean(axis=1)
    if method == "mode":
        df['combined_rank'] = df[[f'{first_score}_rank', 
                                f'{second_score}_rank', 
                                f'{third_score}_rank']
                                ].mode(axis=1)
    else:
        df["combined_rank"] = None

    # Show the plots with normalised scores between 0-1 for the three tests
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    df.plot(kind='line', x='resolutions', y=[f'{first_score}_score_normalized',
                                             f'{second_score}_score_normalized',
                                             f'{third_score}_score_normalized'],
                                             ax=ax)
    
    ax.set_xticks(list(np.arange(df.shape[0])))
    ax.set_xticklabels(df["resolutions"])

    # Add labels and titles
    ax.set_xlabel('Resolutions')
    ax.set_ylabel('scores')
    ax.set_title('Scaled Silhoutte (SH), Inverse Davies Bouldin (DB) and Calinski-Harabasz (CH) scores (0-1; higher is better)')

    # Show the plot
    plt.xticks(rotation=90)
    plt.show()

    #Sort the df based on th ranking with a dict:
    dict_sorting = {"combined_rank":True, "SH_score":False, 
                    "CH_score":False, "DB_score":True}

    # Define the order of the tests and make use of combined rank if true
    if method == "order":
        order_tests = [f'{first_score}_score', 
                       f'{second_score}_score', 
                       f'{third_score}_score']
    else:
        order_tests = ["combined_rank", 
                       f'{first_score}_score', 
                       f'{second_score}_score', 
                       f'{third_score}_score']

    # Retrieve the true-false values of the tests in the specified order
    values_tests = [dict_sorting[key] for key in order_tests]

    # Sort by combined rank, and in case of ties, use silhouette score, calinski harabasz score, and davies bouldin score
    df_sorted = df.sort_values(by=[*order_tests], 
                                   ascending=[*values_tests] #True,
                                   ).reset_index(drop=True)

    # Add final rank based on the sorted order
    df_sorted['final_rank'] = df_sorted.index + 1

    score_columns = ['resolutions',f'{first_score}_score',
                                   f'{second_score}_score', 
                                   f'{third_score}_score', 
                                   'final_rank']
    
    # Display the sorted DataFrame with full ranking
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print("\nRanked clustering resolution table displaying raw test scores:")
    print(df_sorted[score_columns])

    # Extract [res] from '[method]_res_[res]' to use for selection downstream
    df_sorted['resolutions'] = [x.split('_')[2] for x in df_sorted['resolutions']]
    df_sorted['resolutions'] = df_sorted['resolutions'].astype(float)
    df_sorted = df_sorted.round(2)

    # Define the resolution ranges
    range_max_min = resolution_max-resolution_min
    low_resolutions = df_sorted[df_sorted['resolutions'] < round((range_max_min)/3,1)]
    medium_resolutions = df_sorted[(df_sorted['resolutions'] >= round((range_max_min)/3,1)) & 
                                   (df_sorted['resolutions'] < round((range_max_min)/(3/2),1))]
    high_resolutions = df_sorted[df_sorted['resolutions'] >= round((range_max_min)/(3/2),1)]

    # Get the top-ranked resolution for each category
    top_overall = df_sorted.iloc[0]
    top_low = low_resolutions.iloc[0] if not low_resolutions.empty else None
    top_medium = medium_resolutions.iloc[0] if not medium_resolutions.empty else None
    top_high = high_resolutions.iloc[0] if not high_resolutions.empty else None

    # Print the results
    print("\nTop Overall Rank:")
    print(top_overall[score_columns])

    print(f"\nTop Low Clustering Resolution <{round((range_max_min)/3,1)}:")
    if top_low is not None:
        print(top_low[score_columns])
    else:
        print("No low clustering resolutions found.")

    print(f"\nTop Medium Clustering Resolution (>={round((range_max_min)/3,1)} and {round((range_max_min)/(3/2),1)}):")
    if top_medium is not None:
        print(top_medium[score_columns])
    else:
        print("No medium clustering resolutions found.")

    print(f"\nTop High Clustering Resolution (>={round((range_max_min)/(3/2),1)}):")
    if top_high is not None:
        print(top_high[score_columns])
    else:
        print("No high clustering resolutions found.")

    # Convert the float numbers back to original strings
    top_overall = f"{method}_res_{top_overall["resolutions"]:.2f}"
    top_low = f"{method}_res_{top_low["resolutions"]:.2f}"
    top_medium = f"{method}_res_{top_medium["resolutions"]:.2f}"
    top_high = f"{method}_res_{top_high["resolutions"]:.2f}"

    return top_overall, top_low, top_medium, top_high

all, low, medium, high = resolutionrecommender(adata.copy(), tree_columns)
print(all, low, medium, high)

all, low, medium, high = resolutionrecommender(adata2.copy(), columns2, method= "order", test_order = "CH_DB_SH")
print(all, low, medium, high)

    # # plotting
    # fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)

    # df_metrics.plot(kind='line', x='resolutions', y=["calinski_harabasz_score"], 
    #                 ax=ax)
    # ax.set_xticks(list(np.arange(df_metrics.shape[0])))
    # ax.set_xticklabels(df_metrics["resolutions"])


    # # Add labels and titles
    # ax.set_xlabel('Resolutions')
    # ax.set_ylabel('scores')
    # ax.set_title('Calinski Harabasz Score (higher is better)')

    # # Show the plot
    # plt.xticks(rotation=90)
    # plt.show()

    # fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    # df_metrics.plot(kind='line', x='resolutions', y=["silhouette_score",
    #                                                  "davies_bouldin_score"], 
    #                                                  ax=ax)
    # ax.set_xticks(list(np.arange(df_metrics.shape[0])))
    # ax.set_xticklabels(df_metrics["resolutions"])

    # # Add labels and titles
    # ax.set_xlabel('Resolutions')
    # ax.set_ylabel('scores')
    # ax.set_title('Silhoutte (higher is better) and Davies Bouldin (lower is better) scores')

    # # Show the plot
    # plt.xticks(rotation=90)
    # plt.show()