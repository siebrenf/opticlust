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
    test_order="SH_DB_CH",
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
    If "order" is selected then test_order then ranking will be fully based on the test_order parameter.
    :param test_order: if "order" not chosen in rank_method then ranks based on these values only if ties are found.
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

    df_metrics = pd.DataFrame(
        list(zip(sil_list, cal_list, dav_list)),
        columns=["SH_score", "CH_score", "DB_score"],
    )
    df_metrics["resolutions"] = columns
    df_metrics

    df = df_metrics

    # Normalize the scores with min-max scaling (0-1). DB is inverted because lower indicates better clustering.
    df["SH_score_normalized"] = (df["SH_score"] - df["SH_score"].min()) / (
        df["SH_score"].max() - df["SH_score"].min()
    )
    df["CH_score_normalized"] = (df["CH_score"] - df["CH_score"].min()) / (
        df["CH_score"].max() - df["CH_score"].min()
    )
    df["DB_score_normalized"] = 1 - (df["DB_score"] - df["DB_score"].min()) / (
        df["DB_score"].max() - df["DB_score"].min()
    )

    # Rank the test scores
    first_score = test_order.split("_")[0]
    second_score = test_order.split("_")[1]
    third_score = test_order.split("_")[2]

    df[f"{first_score}_rank"] = df[f"{first_score}_score_normalized"].rank(
        ascending=False
    )
    df[f"{second_score}_rank"] = df[f"{second_score}_score_normalized"].rank(
        ascending=False
    )
    df[f"{third_score}_rank"] = df[f"{third_score}_score_normalized"].rank(
        ascending=False
    )

    # Combine the ranks into a median single score
    columns = [f"{first_score}_rank", f"{second_score}_rank", f"{third_score}_rank"]
    df_sub = df[columns]
    print(f"Using rank_method: {rank_method}")
    if rank_method == "median":
        combined_rank = df_sub.median(axis=1)
        df["combined_rank"] = combined_rank
    elif rank_method == "mean":
        combined_rank = df_sub.mean(axis=1)
        df["combined_rank"] = combined_rank
    elif rank_method == "order":
        df["combined_rank"] = None
    else:
        raise ValueError("rank_method must be: median, mean or order")

    # Show the plots with normalised scores between 0-1 for the three tests
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    df.plot(
        kind="line",
        x="resolutions",
        y=[
            f"{first_score}_score_normalized",
            f"{second_score}_score_normalized",
            f"{third_score}_score_normalized",
        ],
        ax=ax,
    )

    ax.set_xticks(list(np.arange(df.shape[0])))
    ax.set_xticklabels(df["resolutions"])

    # Add labels and titles
    ax.set_xlabel("Resolutions")
    ax.set_ylabel("scores")
    ax.set_title(
        "Scaled Silhoutte (SH), Inverse Davies Bouldin (DB) and Calinski-Harabasz (CH) scores (0-1; higher is better)"
    )

    # Show the plot
    plt.xticks(rotation=90)
    plt.show()

    # Sort the df based on th ranking with a dict:
    dict_sorting = {
        "combined_rank": True,
        "SH_score": False,
        "CH_score": False,
        "DB_score": True,
    }

    # Define the order of the tests and make use of combined rank if true
    if rank_method == "order":
        order_tests = [
            f"{first_score}_score",
            f"{second_score}_score",
            f"{third_score}_score",
        ]
    else:
        order_tests = [
            "combined_rank",
            f"{first_score}_score",
            f"{second_score}_score",
            f"{third_score}_score",
        ]

    # Retrieve the true-false values of the tests in the specified order
    values_tests = [dict_sorting[key] for key in order_tests]

    # Sort by combined rank, and in case of ties, use silhouette score, calinski harabasz score, and davies bouldin score
    df_sorted = df.sort_values(
        by=[*order_tests], ascending=[*values_tests]  # True,
    ).reset_index(drop=True)

    # Add final rank based on the sorted order
    df_sorted["final_rank"] = df_sorted.index + 1

    score_columns = [
        "resolutions",
        f"{first_score}_score",
        f"{second_score}_score",
        f"{third_score}_score",
        "final_rank",
    ]

    # Display the sorted DataFrame with full ranking
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print("\nRanked clustering resolution table displaying raw test scores:")
    print(df_sorted[score_columns])

    # Extract [res] from '[method]_res_[res]' to use for selection downstream
    df_sorted["resolutions"] = [x.split("_")[2] for x in df_sorted["resolutions"]]
    df_sorted["resolutions"] = df_sorted["resolutions"].astype(float)
    df_sorted = df_sorted.round(2)

    # Define the resolution ranges
    range_max_min = resolution_max - resolution_min
    low_resolutions = df_sorted[
        df_sorted["resolutions"] < round((range_max_min) / 3, 1)
    ]
    medium_resolutions = df_sorted[
        (df_sorted["resolutions"] >= round((range_max_min) / 3, 1))
        & (df_sorted["resolutions"] < round((range_max_min) / (3 / 2), 1))
    ]
    high_resolutions = df_sorted[
        df_sorted["resolutions"] >= round((range_max_min) / (3 / 2), 1)
    ]

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

    print(
        f"\nTop Medium Clustering Resolution (>={round((range_max_min)/3,1)} and {round((range_max_min)/(3/2),1)}):"
    )
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
    top_overall = f"{method_clustering}_res_{top_overall["resolutions"]:.2f}"
    top_low = f"{method_clustering}_res_{top_low["resolutions"]:.2f}"
    top_medium = f"{method_clustering}_res_{top_medium["resolutions"]:.2f}"
    top_high = f"{method_clustering}_res_{top_high["resolutions"]:.2f}"

    return top_overall, top_low, top_medium, top_high
