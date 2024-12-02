import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


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
    if "pca" not in adata.uns.keys():
        raise RuntimeError("A PCA is required!")
    if cluster_kwargs is None:
        cluster_kwargs = {}
        if method == "leiden":
            cluster_kwargs["flavor"] = "igraph"

    columns = []
    resolutions = np.linspace(
        resolution_min, resolution_max, num=samples, endpoint=True
    ).tolist()
    for res in resolutions:
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


def clustering_plot(
    adata,
    columns,
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
    if columns[0].count("_") != 2:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    method = columns[0].split("_", 1)[0]
    if method not in ["leiden", "louvain"]:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")

    try:
        x = sorted([float(c.rsplit("_", 1)[1]) for c in columns])
    except ValueError:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    y = [len(adata.obs[c].unique()) for c in columns]

    # for each number of clusters, store the resolutions
    clust = {}
    for i in range(len(y)):
        c = y[i]
        if c not in clust:
            clust[c] = []
        clust[c].append(x[i])

    # for each number of clusters, store the mean and median resolution
    x_clust_med = []
    y_clust_med = []
    x_clust_mean = []
    y_clust_mean = []
    for c in sorted(clust):
        if c == 1:  # a single cluster is not informative
            continue
        resolutions = clust[c]
        # When many resolutions yield the same number of clusters,
        # this can be considered a "stable" clustering.
        # To reduce downstream analysis complexity, we can filter out
        # any "unstable" clustering.
        if x_clust_med and len(resolutions) < min_n_resolutions:
            continue

        x_med = nearest(np.median(resolutions), resolutions)
        y_med = c
        if x_clust_med and x_med < x_clust_med[-1]:
            # We expect the the cluster resolution to increase with number of clusters.
            # If this does not happen, skip the resolution with on the smallest
            # sample size.
            if len(resolutions) > len(clust[y_clust_med[-1]]):
                x_clust_med[-1] = x_med  # skip the previous resolution
                y_clust_med[-1] = y_med
            else:
                pass  # skip this resolution
        else:
            x_clust_med.append(x_med)
            y_clust_med.append(y_med)

        x_mean = nearest(np.mean(resolutions), resolutions)
        y_mean = c
        if x_clust_mean and x_mean < x_clust_mean[-1]:
            # see above for explanation
            if len(resolutions) > len(clust[y_clust_mean[-1]]):
                x_clust_mean[-1] = x_mean  # skip the previous resolution
                y_clust_mean[-1] = y_mean
            else:
                pass  # skip this resolution
        else:
            x_clust_mean.append(x_mean)
            y_clust_mean.append(y_mean)

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

    # 3 & 4) plot the mean and median resolution at each number of clusters
    #   zorder: mean line > median line
    #   zorder: median scatter > mean scatter
    ax.scatter(x_clust_mean, y_clust_mean, c="C1", alpha=1, zorder=-8)
    ax.plot(
        x_clust_mean,
        y_clust_mean,
        c="C1",
        ls="--",
        zorder=-5,
        label=f"mean resolution",
    )
    ax.plot(
        x_clust_med,
        y_clust_med,
        c="C0",
        zorder=-6,
        label=f"median resolution",
    )
    for cx, cy in zip(x_clust_med, y_clust_med):
        ax.scatter(
            cx,
            cy,
            c="C0",
            zorder=-7,
            label=f"n={cy: >2} res={cx:4.2f}",
        )

    ax.grid(which="major")
    ax.set_title(
        f"Number of clusters over {lc} {method.capitalize()} clustering resolutions"
    )
    ax.set_xlabel(f"{method.capitalize()} clustering resolution")
    ax.set_ylabel("Number of clusters")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.7)
    plt.show()

    # return the median resolution per number of cluster
    cluster_resolutions = []
    for res, n_clusters in zip(x_clust_med, y_clust_med):
        cluster_resolutions.append(f"{method}_res_{res:4.2f}")

    if return_plot:
        return cluster_resolutions, fig, ax
    return cluster_resolutions


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def nearest(val, vals):
    best = -1, float("inf")
    for i in range(len(vals)):
        diff = abs(val - vals[i])
        if diff < best[1]:
            best = i, diff
    return vals[best[0]]
