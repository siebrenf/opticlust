import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from natsort import natsort_keygen, natsorted
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from tqdm.auto import tqdm


def score_resolutions(
    adata,
    columns,
    tests="SH_CH_DB",
    method="mean",
    max_n_silhouette=50_000,
    figsize=(16, 8),
    subplot_kwargs=None,
    return_plot=False,
):
    """
    Score clustering resolutions based on multiple clustering metrics.
    Silhouette (SH), Davies Bouldin (DB) and Calinski-Harabasz (CH) scores are all calculated.
    Updates adata.uns["opticlust"] with scores per resolution.

    :param adata: dataset with precalculated UMAP and clusters.
    :param columns: list of adata.obs column names to use in the plot.
    :param tests: which metrics are included in the combined score and rank
      (options: SH, DB, CH, underscore separated. Default: "SH_DB_CH").
    :param method: combines scores from tests (options: "median", "mean", "order").
    If "order" is selected the scores are ranked in order.
    With all options, the order of parameter tests is used as tiebreaker.
    :param max_n_silhouette: subset cells for the Silhouette score to this number. Use -1 for all cells.
    :param figsize: matplotlib figsize.
    :param subplot_kwargs: kwargs passed on to plt.subplot.
    :param return_plot: if True, also returns fig and ax.
    """
    use_subset = max_n_silhouette != -1 and len(adata.obs) > max_n_silhouette

    columns = natsorted(columns)
    if columns[0].count("_") != 2:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    method_clustering = columns[0].split("_", 1)[0]
    if method_clustering not in ["leiden", "louvain"]:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")

    plotdf = sc.get.obs_df(
        adata, keys=[*columns], obsm_keys=[("X_umap", 0), ("X_umap", 1)]
    )
    if use_subset:
        plotdf_subset = plotdf.sample(max_n_silhouette, random_state=42)
    dim1 = plotdf["X_umap-0"].to_numpy()
    dim2 = plotdf["X_umap-1"].to_numpy()
    dims = np.concatenate((dim1.reshape(-1, 1), dim2.reshape(-1, 1)), axis=1)

    sil_list = []
    cal_list = []
    dav_list = []
    for i in tqdm(columns):
        test_res = plotdf[i].to_numpy()
        try:
            test_res2 = plotdf_subset[i].to_numpy() if use_subset else test_res
            sil_list.append(silhouette_score(dims, test_res2))
        except (ValueError, AttributeError):
            sil_list.append(np.nan)
        try:
            cal_list.append(calinski_harabasz_score(dims, test_res))
        except (ValueError, AttributeError):
            cal_list.append(np.nan)
        try:
            dav_list.append(davies_bouldin_score(dims, test_res))
        except (ValueError, AttributeError):
            dav_list.append(np.nan)
    df = pd.DataFrame(
        list(zip(sil_list, cal_list, dav_list)),
        columns=["SH_score", "CH_score", "DB_score"],
        index=columns,
    )

    # Normalize the scores with min-max scaling (0-1).
    # DB is inverted because lower indicates better clustering.
    df["SH_score_normalized"] = (df["SH_score"] - df["SH_score"].min()) / (
        df["SH_score"].max() - df["SH_score"].min()
    )
    df["CH_score_normalized"] = (df["CH_score"] - df["CH_score"].min()) / (
        df["CH_score"].max() - df["CH_score"].min()
    )
    df["DB_score_normalized"] = 1 - (df["DB_score"] - df["DB_score"].min()) / (
        df["DB_score"].max() - df["DB_score"].min()
    )

    # Combine the scores
    columns = [f"{score}_score_normalized" for score in tests.split("_")]
    if method == "median":
        df["combined_score_normalized"] = df[columns].median(axis=1)
    elif method == "mean":
        # TODO: does a weighted average make sense?
        score_weights = {"CH": 1.0, "DB": 1.0, "SH": 1.0}
        df2 = df[columns].copy()
        for score in tests.split("_"):
            df2[f"{score}_score_normalized"] = (
                df2[f"{score}_score_normalized"] * score_weights[score]
            )
        df["combined_score_normalized"] = df2[columns].mean(axis=1)
    elif method == "order":
        # Add each test to the combined score,
        #  dividing each test by a larger number to act as tiebreaker .
        df["combined_score_normalized"] = 0
        for i, score in enumerate(tests.split("_")):
            df["combined_score_normalized"] += df[f"{score}_score_normalized"] / (
                1000**i
            )
        # Re-scale the score
        col = "combined_score_normalized"
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    else:
        raise ValueError("method must be: median, mean or order")

    # Rank the cluster resolutions, using successive scores in param tests as tiebreaker
    # First, define the metrics to sort by, and their order
    order_tests = [f"{score}_score" for score in tests.split("_")]
    if method != "order":
        order_tests = ["combined_score_normalized"] + order_tests

    # Second, determine if the sorting is ascending/descending per metric
    dict_sorting = {
        "combined_score_normalized": False,  # higher is better
        "SH_score": False,  # higher is better
        "CH_score": False,  # higher is better
        "DB_score": True,  # lower is better
    }
    values_tests = [dict_sorting[key] for key in order_tests]

    # Finally, sort and add the rank based on the sorted order
    df.sort_values(by=[*order_tests], ascending=[*values_tests], inplace=True)
    df["rank"] = df.reset_index().index + 1

    # Add the metrics to adata
    adata.uns["opticlust"] = df.sort_index(key=natsort_keygen())
    adata.uns["opticlust_params"] = {
        "INFO": "This dict contains the parameters used to generate adata.uns['opticlust']",
        "columns": columns,
        "tests": tests,
        "method": method,
    }

    return _plot_metrics(
        df,
        method_clustering,
        tests,
        method,
        figsize,
        subplot_kwargs,
        return_plot,
    )


def _plot_metrics(
    df,
    method_clustering,
    tests,
    method,
    figsize=(16, 8),
    subplot_kwargs=None,
    return_plot=False,
):
    if subplot_kwargs is None:
        subplot_kwargs = {}
    df.sort_index(inplace=True)

    # Show the plots with normalised scores between 0-1 for the three tests
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    df.plot(
        kind="line",
        ls="-",
        y=[
            "SH_score_normalized",
            "DB_score_normalized",
            "CH_score_normalized",
        ],
        ax=ax,
    )
    df.plot(
        kind="line",
        ls="--",
        y="combined_score_normalized",
        ax=ax,
    )

    # Add vertical bands
    for i in range(len(df)):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.5, color="lightgrey", zorder=-1)

    # Add labels and titles
    ax.set_xticks(
        ticks=list(range(df.shape[0])),
        labels=[x.split("_")[2] for x in df.index],
        rotation=90,
    )
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_xlabel(f"{method_clustering.capitalize()} clustering resolution")
    ax.set_ylabel("Metric scores")
    ax.set_title(
        f"Metric scores per {method_clustering.capitalize()} clustering resolution"
        "\n (normalized & scaled between 0-1; higher is better)"
    )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    d = {
        "SH": "Scaled Silhouette",
        "DB": "Inverse Davies Bouldin",
        "CH": "Calinski-Harabasz",
        "combined": f"Combined ({tests} {method})",
    }
    labels = [d.get(lbl.split("_")[0], lbl) for lbl in labels]
    ax.legend(
        title="Metrics",
        handles=handles,
        labels=labels,
    )

    plt.tight_layout()
    plt.show()
    if return_plot:
        return fig, ax


def recommend_resolutions(
    adata,
    columns=None,
    resolution_min=None,
    resolution_max=None,
):
    """
    Recommends clustering resolutions based on scores of multiple clustering related tests.
    Silhouette (SH), Davies Bouldin (DB) and Calinski-Harabasz (CH) scores are all calculated.

    :param adata: dataset
    :param columns: the columns to recommend from.
    :param resolution_min: the lowest clustering resolution.
    :param resolution_max: the highest clustering resolution.
    """
    tests = adata.uns["opticlust_params"]["tests"]
    score_columns = [f"{score}_score" for score in tests.split("_")] + [
        "rank",
    ]
    df = adata.uns["opticlust"][score_columns].copy()
    if columns is None:
        columns = df.index.to_list()
    else:
        if len(set(columns)) != len(set(columns) & set(df.index)):
            raise IndexError(
                "Not all given columns found. Please run score_resolutions() with these columns!"
            )
        df = df.loc[list(set(columns))]
    df.sort_values("rank", inplace=True)

    method_clustering = columns[0].split("_", 1)[0]
    if method_clustering not in ["leiden", "louvain"]:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")

    # Display the sorted DataFrame with full ranking
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print("\nRanked clustering resolution table displaying raw test scores:")
    print(df)

    # Separate the resolution into 3 bins, and return the top resolution of each.

    # Extract [res] from '[method]_res_[res]' to use for selection downstream
    df["resolutions"] = [x.split("_")[2] for x in df.index]
    df["resolutions"] = df["resolutions"].astype(float)
    df = df.round(2).dropna()

    # Define the resolution ranges
    if resolution_max is None:
        resolution_max = df["resolutions"].max()
    if resolution_min is None:
        resolution_min = df["resolutions"].min()
    range_max_min = resolution_max - resolution_min
    low_resolutions = df[df["resolutions"] < round(range_max_min / 3, 1)]
    medium_resolutions = df[
        (df["resolutions"] >= round(range_max_min / 3, 1))
        & (df["resolutions"] < round(range_max_min / (3 / 2), 1))
    ]
    high_resolutions = df[df["resolutions"] >= round(range_max_min / (3 / 2), 1)]

    # Get the top-ranked resolution for each category
    top_overall = df.iloc[0]
    top_low = low_resolutions.iloc[0] if not low_resolutions.empty else None
    top_medium = medium_resolutions.iloc[0] if not medium_resolutions.empty else None
    top_high = high_resolutions.iloc[0] if not high_resolutions.empty else None

    # Print the results
    print("\nTop Overall Rank:")
    print(top_overall)

    print(f"\nTop Low Clustering Resolution <{round(range_max_min / 3, 1)}:")
    if top_low is not None:
        print(top_low)
    else:
        print("No low clustering resolutions found.")

    print(
        f"\nTop Medium Clustering Resolution (>={round(range_max_min / 3, 1)} and {round(range_max_min / (3 / 2), 1)}):"
    )
    if top_medium is not None:
        print(top_medium)
    else:
        print("No medium clustering resolutions found.")

    print(f"\nTop High Clustering Resolution (>={round(range_max_min / (3 / 2), 1)}):")
    if top_high is not None:
        print(top_high)
    else:
        print("No high clustering resolutions found.")

    # Convert the float numbers back to original strings
    top_overall = f"{method_clustering}_res_{top_overall['resolutions']:.2f}"
    top_low = f"{method_clustering}_res_{top_low['resolutions']:.2f}"
    top_medium = f"{method_clustering}_res_{top_medium['resolutions']:.2f}"
    top_high = f"{method_clustering}_res_{top_high['resolutions']:.2f}"

    return top_overall, top_low, top_medium, top_high
