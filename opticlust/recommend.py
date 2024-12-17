import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def resolutionrecommender(
    adata,
    columns,
    resolution_min=0.0,
    resolution_max=2.0,
    rank_method="median",
    tests="SH_CH_DB",
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
    :param rank_method: combines ranked scores from tests with the options "median", "mean" or "order".
    If "order" is selected then the selected tests will be ranked in order.
    :param tests: if "order" not chosen in rank_method then ranks based on these values only if ties are found.
    All possible combinations can be chosen, including: SH_CH_DB, DB_SH_CH etc. (default SH_DB_CH).
    :param figsize: matplotlib figsize
    :param subplot_kwargs: kwargs passed on to plt.subplot
    :param return_plot: if True, also returns fig and ax
    """

    if subplot_kwargs is None:
        subplot_kwargs = {}

    if columns[0].count("_") != 2:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    method_clustering = columns[0].split("_", 1)[0]
    if method_clustering not in ["leiden", "louvain"]:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")

    plotdf = sc.get.obs_df(
        adata, keys=[*columns], obsm_keys=[("X_umap", 0), ("X_umap", 1)]
    )
    dim1 = plotdf["X_umap-0"].to_numpy()
    dim2 = plotdf["X_umap-1"].to_numpy()
    dims = np.concatenate((dim1.reshape(-1, 1), dim2.reshape(-1, 1)), axis=1)

    sil_list = []
    cal_list = []
    dav_list = []
    for i in columns:
        test_res = plotdf[i].to_numpy()
        try:
            sil_list.append(silhouette_score(dims, test_res))
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
    if rank_method == "median":
        df["combined_score_normalized"] = df[columns].median(axis=1)
    elif rank_method == "mean":
        # TODO: does a weighted average make sense?
        score_weights = {"CH": 1.0, "DB": 1.0, "SH": 1.0}
        df2 = df[columns].copy()
        for score in tests.split("_"):
            df2[f"{score}_score_normalized"] = (
                df2[f"{score}_score_normalized"] * score_weights[score]
            )
        df["combined_score_normalized"] = df2[columns].mean(axis=1)
    elif rank_method == "order":
        # TODO: scale output
        df["combined_score_normalized"] = 0
        for i, score in enumerate(tests.split("_")):
            df["combined_score_normalized"] += df[f"{score}_score_normalized"] / (
                1000**i
            )
    else:
        raise ValueError("rank_method must be: median, mean or order")

    _plot_metrics(
        df,
        method_clustering,
        figsize,
        subplot_kwargs,
    )

    # Sort the cluster resolutions, using next score as tiebreaker
    # First, define the metrics to sort by, and their order
    order_tests = [f"{score}_score" for score in tests.split("_")]
    if rank_method != "order":
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
    df_sorted = df.sort_values(by=[*order_tests], ascending=[*values_tests])
    df_sorted["rank"] = df_sorted.reset_index().index + 1

    # Add the metrics to adata
    adata.uns["opticlust"] = df_sorted.sort_index()
    print("Metrics have been stored under adata.uns['opticlust']")

    # Display the sorted DataFrame with full ranking
    score_columns = [f"{score}_score" for score in tests.split("_")] + [
        "combined_score_normalized",
        "rank",
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print("\nRanked clustering resolution table displaying raw test scores:")
    print(df_sorted[score_columns])

    return _recommend_resolutions(
        df_sorted, resolution_max, resolution_min, score_columns, method_clustering
    )


def _plot_metrics(
    df,
    method_clustering,
    figsize,
    subplot_kwargs,
):
    # Show the plots with normalised scores between 0-1 for the three tests
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    df.plot(
        kind="line",
        y=[
            "SH_score_normalized",
            "DB_score_normalized",
            "CH_score_normalized",
            "combined_score_normalized",
        ],
        ax=ax,
    )
    for i in range(len(df)):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.5, color="lightgrey", zorder=-1)

    ax.set_xticks(list(np.arange(df.shape[0])))
    ax.set_xticklabels([x.split("_")[2] for x in df.index])

    # Add labels and titles
    plt.xticks(rotation=90)
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
    }
    labels = [d.get(l.split("_")[0], l) for l in labels]
    ax.legend(
        title="Metrics",
        handles=handles,
        labels=labels,
    )

    # Show the plot
    plt.show()


def _recommend_resolutions(
    df_sorted, resolution_max, resolution_min, score_columns, method_clustering
):
    # Extract [res] from '[method]_res_[res]' to use for selection downstream
    df_sorted["resolutions"] = [x.split("_")[2] for x in df_sorted.index]
    df_sorted["resolutions"] = df_sorted["resolutions"].astype(float)
    df_sorted = df_sorted.round(2)

    # Define the resolution ranges
    range_max_min = resolution_max - resolution_min
    low_resolutions = df_sorted[df_sorted["resolutions"] < round(range_max_min / 3, 1)]
    medium_resolutions = df_sorted[
        (df_sorted["resolutions"] >= round(range_max_min / 3, 1))
        & (df_sorted["resolutions"] < round(range_max_min / (3 / 2), 1))
    ]
    high_resolutions = df_sorted[
        df_sorted["resolutions"] >= round(range_max_min / (3 / 2), 1)
    ]

    # Get the top-ranked resolution for each category
    top_overall = df_sorted.iloc[0]
    top_low = low_resolutions.iloc[0] if not low_resolutions.empty else None
    top_medium = medium_resolutions.iloc[0] if not medium_resolutions.empty else None
    top_high = high_resolutions.iloc[0] if not high_resolutions.empty else None

    # Print the results
    print("\nTop Overall Rank:")
    print(top_overall[score_columns])

    print(f"\nTop Low Clustering Resolution <{round(range_max_min / 3, 1)}:")
    if top_low is not None:
        print(top_low[score_columns])
    else:
        print("No low clustering resolutions found.")

    print(
        f"\nTop Medium Clustering Resolution (>={round(range_max_min / 3, 1)} and {round(range_max_min / (3 / 2), 1)}):"
    )
    if top_medium is not None:
        print(top_medium[score_columns])
    else:
        print("No medium clustering resolutions found.")

    print(f"\nTop High Clustering Resolution (>={round(range_max_min / (3 / 2), 1)}):")
    if top_high is not None:
        print(top_high[score_columns])
    else:
        print("No high clustering resolutions found.")

    # Convert the float numbers back to original strings
    top_overall = f"{method_clustering}_res_{top_overall['resolutions']:.2f}"
    top_low = f"{method_clustering}_res_{top_low['resolutions']:.2f}"
    top_medium = f"{method_clustering}_res_{top_medium['resolutions']:.2f}"
    top_high = f"{method_clustering}_res_{top_high['resolutions']:.2f}"

    return top_overall, top_low, top_medium, top_high
