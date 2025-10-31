import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from natsort import natsorted
from tqdm.auto import tqdm

from .utils import validate_resolutions


def clustering(
    adata,
    resolution_min=0.0,
    resolution_max=2.0,
    samples=81,
    method="leiden",
    cluster_kwargs=None,
):
    """
    Clustering the dataset at multiple resolutions

    :param adata: dataset
    :param resolution_min: the lowest clustering resolution
    :param resolution_max: the highest clustering resolution
    :param samples: number of clusterings between the min and max resolutions
    :param method: clustering methods. Options: "leiden" or "louvain"
    :param cluster_kwargs: kwargs passed on the cluster function
    :return: columns: list of column names generated in adata.obs
    """
    if cluster_kwargs is None:
        cluster_kwargs = {}
        if method == "leiden":
            cluster_kwargs["flavor"] = "igraph"

    columns = []
    if resolution_min < 0:
        raise ValueError("resolution_min must be non-negative")
    if resolution_max >= 10:
        raise ValueError("resolution_max must be below 10")
    resolutions = np.linspace(
        resolution_min, resolution_max, num=samples, endpoint=True
    ).tolist()
    for res in tqdm(resolutions, unit=" cluster resolutions"):
        key = f"{method}_res_{res:4.2f}"
        columns.append(key)
        if key in adata.obs.columns:
            continue
        if method == "leiden":
            sc.tl.leiden(adata, key_added=key, resolution=res, **cluster_kwargs)
        elif method == "louvain":
            sc.tl.louvain(adata, key_added=key, resolution=res, **cluster_kwargs)
        else:
            raise ValueError(f"method can only be leiden or louvain, not {method}")
    return columns


METHOD2COLOR = {
    "mean": "C0",  # blue
    "median": "C3",  # red
    "middle": "C1",  # yellow
    "score": "C2",  # green
}
METHOD2LS = {
    "mean": ":",
    "median": "-",
    "middle": "--",
    "score": "-.",
}


def clustering_plot(
    adata,
    columns,
    method="middle",
    min_n_resolutions=1,
    window_size=5,
    figsize=(16, 8),
    subplot_kwargs=None,
    return_plot=False,
):
    """
    Plot the effect of clustering resolution on the number of clusters identified.
    Returns the median resolution for each number of clusters.

    :param adata: dataset
    :param columns: list of adata.obs column names to use in the plot.
    Column names must be in the shape "[method]_res_[res]".
    :param method: resolution selection method (options: "mean", "median", "middle", "score").
    :param min_n_resolutions: filters the list of representative cluster resolutions
     by a minimum number of resolutions yielding the same number of clusters.
    :param window_size: width of the moving window.
    :param figsize: matplotlib figsize
    :param subplot_kwargs: kwargs passed on to plt.subplot
    :param return_plot: if True, also returns fig and ax

    :return: cluster_resolutions: a list of representative cluster resolutions,
    matching column names in adata.obs.
    """
    if subplot_kwargs is None:
        subplot_kwargs = {}
    n = window_size
    lc = len(columns)
    if method == "score" and "opticlust" not in adata.uns:
        raise ValueError("Please run score_resolutions() first!")
    columns = natsorted(columns)
    method_clustering, x = validate_resolutions(columns)
    y = [len(adata.obs[c].unique()) for c in columns]

    # for each number of clusters, store the resolutions
    clust = {}
    for resolution, n_clusters in zip(x, y):
        if n_clusters not in clust:
            clust[n_clusters] = []
        clust[n_clusters].append(resolution)

    # for each number of clusters, compute interesting resolutions
    y_clust = []
    x_clust_med = []
    x_clust_mean = []
    x_clust_rank = []
    x_clust_mid = []
    for n_clusters in sorted(clust):
        resolutions = clust[n_clusters]
        # When many resolutions yield the same number of clusters,
        # this can be considered a "stable" clustering.
        # To reduce downstream analysis complexity, we can filter out
        # any "unstable" clustering.
        if len(resolutions) < min_n_resolutions:
            continue

        y_clust.append(n_clusters)
        x_clust_med.append(nearest(np.median(resolutions), resolutions))
        x_clust_mean.append(nearest(np.mean(resolutions), resolutions))

        # use the metrics from score_resolutions() to select
        #  the top scoring resolution per n clusters
        if "opticlust" in adata.uns:
            res = [f"{method_clustering}_res_{r:.2f}" for r in resolutions]
            res = adata.uns["opticlust"].loc[res]["rank"].sort_values().index[0]
            x_clust_rank.append(float(res.split("_")[2]))

        # use the middle resolution from the longest consecutive sequence of resolutions
        res = longest_consecutive_subsequence(resolutions, x)
        x_clust_mid.append(nearest(np.median(res), res))

    # collect the selected resolutions per number of cluster
    if method == "score":
        x_clust = x_clust_rank
    elif method == "mean":
        x_clust = x_clust_mean
    elif method == "median":
        x_clust = x_clust_med
    elif method == "middle":
        x_clust = x_clust_mid
    else:
        raise ValueError("method must be 'mean', 'median', 'middle', 'score'!")
    cluster_resolutions = []
    for res, n_clusters in zip(x_clust, y_clust):
        if n_clusters > 1:  # a single cluster is not informative
            cluster_resolutions.append(f"{method_clustering}_res_{res:4.2f}")

    # plotting
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)

    # 1) plot all resolutions
    ax.scatter(x, y, color="grey", marker="o", alpha=1.0, zorder=-10)

    # 2) plot the moving average of all resolutions
    x_avg = x[(n - 1) // 2 : -(n - 1) // 2]
    y_avg = moving_average(y, n=n)
    ax.plot(
        x_avg,
        y_avg,
        color="black",
        zorder=-9,
        label=f"moving average (w={n})",
    )

    # 3, 4 & 5) plot the mean, median and middle resolution at each number of clusters
    #   zorder: mean line > median line
    #   zorder: median scatter > mean scatter
    ax.scatter(x_clust_mean, y_clust, c=METHOD2COLOR["mean"], alpha=1, zorder=-8)
    ax.plot(
        x_clust_mean,
        y_clust,
        c=METHOD2COLOR["mean"],
        ls=METHOD2LS["mean"],
        zorder=-5,
        label="mean resolution",
    )
    # ax.scatter(x_clust_med, y_clust, c=METHOD2COLOR["median"], alpha=1, zorder=-6)
    # ax.plot(
    #     x_clust_med,
    #     y_clust,
    #     c=METHOD2COLOR["median"],
    #     ls=METHOD2LS["median"],
    #     zorder=-6,
    #     label="median resolution",
    # )
    ax.scatter(x_clust_mid, y_clust, c=METHOD2COLOR["middle"], alpha=1, zorder=-6)
    ax.plot(
        x_clust_mid,
        y_clust,
        c=METHOD2COLOR["middle"],
        ls=METHOD2LS["middle"],
        zorder=-6,
        label="middle resolution\n (longest consecutive sequence)",
    )

    # 6) plot the best scoring resolutions
    if "opticlust" in adata.uns:
        ax.scatter(x_clust_rank, y_clust, c=METHOD2COLOR["score"], alpha=1, zorder=-6)
        ax.plot(
            x_clust_rank,
            y_clust,
            c=METHOD2COLOR["score"],
            ls=METHOD2LS["score"],
            zorder=-4,
            label="best scoring resolution",
        )

    # 7) add the selected resolutions to the legend
    for cx, cy in zip(x_clust, y_clust):
        ax.scatter(
            cx,
            cy,
            c=METHOD2COLOR[method],
            zorder=-10,
            label=f"n={cy: >2} res={cx:4.2f}",
        )

    # 8) layout
    ax.grid(which="major")
    ax.set_title(
        f"Number of clusters over {lc} {method_clustering.capitalize()} clustering resolutions"
    )
    ax.set_xlabel(f"{method_clustering.capitalize()} clustering resolution")
    ax.set_ylabel("Number of clusters")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.7)
    plt.tight_layout()
    if return_plot:
        return cluster_resolutions, fig, ax
    else:
        plt.show()
        return cluster_resolutions


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def nearest(target, resolutions):
    best = 0, float("inf")
    for res in resolutions:
        diff = abs(target - res)
        if diff < best[1]:
            best = res, diff
    return best[0]


def longest_consecutive_subsequence(subset_resolutions, all_resolutions):
    longest = []
    current = []
    # look through all_resolutions (within bounds)
    # grow a list of current consecutive resolutions every time a match is found
    # when a mismatch is found, keep the longest list, and reset the current list
    start = all_resolutions.index(subset_resolutions[0])
    end = all_resolutions.index(subset_resolutions[-1])
    for res in all_resolutions[start : end + 1]:
        if res in subset_resolutions:
            current.append(res)
        else:
            # tie-breaker with equal lengths: lower resolution
            if len(current) > len(longest):
                longest = current
            current = []
    if len(current) > len(longest):
        longest = current
    return longest
