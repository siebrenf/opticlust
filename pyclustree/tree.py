import warnings

import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc


def clustree(
    adata, cluster_resolutions, rename_cluster=True, cluster2color=None, colors=None
):
    """
    Map the clusters to a tree structure.
    The resolution descends along the y-axis (min at the top, max at the bottom),
    and the clusters are distributed over the x-axis by size.
    Daughter clusters are plotted proximal to the parent with the highest cell overlap.

    If rename_cluster is True, clusters are updated in adata.obs and colors are updated in adata.uns
    such that the color is passed from parent the daughter with the largest cell overlap.
    Note that this does **not** alter the clusters or data in any way.

    :param adata: dataset
    :param cluster_resolutions: a dict with number of clusters as key, and an adata.obs column as value.
    :param rename_cluster: change the cluster names in adata.obs
    :param cluster2color: optional dictionary to specify the cluster to color relation
    :param colors: optional list of colors (default: scanpy defaults)
    :return: plot_data: a dictionary used in clustree_plot
    """
    n_clusters = max(cluster_resolutions.keys())
    n_resolutions = len([n for n in cluster_resolutions.keys() if n != 1])
    n_cells = len(adata.obs)

    g = nx.DiGraph()
    y_ticks = []  # plot label
    y_labels = []  # plot label
    column = cluster_resolutions[n_clusters]
    if column.count("_") != 2:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    method = column.split("_", 1)[0]
    if method not in ["leiden", "louvain"]:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    try:
        float(column.rsplit("_", 1)[1])
    except ValueError:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")

    if cluster2color is None:
        # set cluster colors
        cluster2color = {}
        if colors is None:
            colors = sc.pl.palettes.vega_20_scanpy  # default scanpy colors
        column = cluster_resolutions[n_clusters]
        for i, c in enumerate(adata.obs[column].cat.categories):
            # repeat colors if there are more clusters than colors
            while i > len(colors):
                i -= len(colors)
            cluster2color[c] = colors[i]
    elif colors:
        warnings.warn(
            "You provided both cluster2color and colors. "
            "Argument colors will be ignored."
        )

    cluster2barcodes = {}
    r_prev = None
    y = n_resolutions - 1  # plot coordinate
    for n_clusters_, column in cluster_resolutions.items():
        if n_clusters_ == 1:
            continue

        r = column.rsplit("_", 1)[1]  # r = resolution, e.g. "0.30"
        y_ticks.append(y)
        y_labels.append(r)
        cluster2barcodes[r] = {}
        for c in adata.obs[column].unique().categories:  # c = cluster name, e.g. "3"
            barcodes = set(adata.obs[adata.obs[column] == c].index)
            cluster2barcodes[r][c] = barcodes

        x = 0  # plot coordinate
        if r_prev is None:

            # sort clusters by size (descending)
            cluster2barcodes[r] = dict(
                sorted(
                    cluster2barcodes[r].items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )
            )
            for c, barcodes in cluster2barcodes[r].items():
                name = f"{r}_{c}"
                size_node = len(barcodes)
                dx = (size_node / n_cells) * n_clusters
                x_node = x + dx / 2
                y_node = y
                color_node = cluster2color[c]
                g.add_node(name, x=x_node, y=y_node, size=size_node, color=color_node)

                x = x + dx

        else:

            # sort cluster by overlap with parent clusters (descending)
            order = []
            for c_prev, barcodes_prev in cluster2barcodes[r_prev].items():
                name_prev = f"{r_prev}_{c_prev}"

                # collect the overlap of this parent cluster with all cluster at the present resolution
                edges = {}
                for c in cluster2barcodes[r]:
                    barcodes = cluster2barcodes[r][c]
                    overlap = len(barcodes & barcodes_prev)
                    if overlap:  # skip 0 overlap
                        edges[c] = overlap
                # and sort by overlap (descending)
                edges = dict(
                    sorted(edges.items(), key=lambda item: item[1], reverse=True)
                )

                for c, overlap in edges.items():
                    name = f"{r}_{c}"
                    g.add_edge(name_prev, name, size=overlap)
                    if c not in order:
                        order.append(c)

                        barcodes = cluster2barcodes[r][c]
                        size_node = len(barcodes)
                        dx = (size_node / n_cells) * n_clusters
                        x_node = x + dx / 2
                        y_node = y
                        color_node = cluster2color[c]
                        g.add_node(
                            name, x=x_node, y=y_node, size=size_node, color=color_node
                        )

                        x = x + dx

            # apply sorting
            cluster2barcodes[r] = {c: cluster2barcodes[r][c] for c in order}

        y -= 1
        r_prev = r

    if rename_cluster:
        _clustering_rename(adata, g, cluster2barcodes, cluster2color, method)

    # plotting needs a bunch of variables, so let's wrap them up nicely
    plot_data = {
        "graph": g,
        "dimensions": {
            "n_resolutions": n_resolutions,
            "n_clusters": n_clusters,
        },
        "axis": {
            "y_ticks": y_ticks,
            "y_labels": y_labels,
        },
        "cluster_method": method,
    }
    return plot_data


def _clustering_rename(adata, g, cluster2barcodes, cluster2color, method):
    """rename the clusters adata.obs, such that the clusters names remain consistent over multiple resolutions"""
    rename_dict = {}
    r_prev = None
    for r in cluster2barcodes.keys():
        rename_dict[r] = {}
        if r_prev is None:
            for c in cluster2barcodes[r].keys():
                rename_dict[r][c] = f"c{c}"  # no changes
        else:
            # sort clusters by overlap with previous clusters (descending)
            overlaps = {}
            tmp = dict(
                sorted(
                    cluster2barcodes[r_prev].items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )
            )
            for c_prev, barcodes_prev in tmp.items():
                overlaps[c_prev] = {}
                for c, barcodes in cluster2barcodes[r].items():
                    overlap = len(barcodes & barcodes_prev)
                    overlaps[c_prev][c] = overlap
                overlaps[c_prev] = dict(
                    sorted(
                        overlaps[c_prev].items(), key=lambda item: item[1], reverse=True
                    )
                )

            seen = set()
            seen2 = set()
            while len(rename_dict[r]) != len(cluster2barcodes[r]):
                for c_prev in overlaps.keys():
                    for c in list(overlaps[c_prev]):
                        if c in seen:
                            continue
                        seen.add(c)
                        if c_prev in seen2:
                            c_prev = max([int(i) for i in seen2]) + 1
                        seen2.add(c_prev)
                        if c_prev in rename_dict[r_prev]:
                            rename_dict[r][c] = rename_dict[r_prev][c_prev]
                        else:
                            # add a temporary prefix "c" here
                            # so the clusters aren't merged by pandas.replace
                            rename_dict[r][c] = f"c{c_prev}"
                        break

        r_prev = r

    # rename the clusters in adata
    for r, d in rename_dict.items():
        column = f"{method}_res_{r}"
        adata.obs[column] = adata.obs[column].cat.rename_categories(d)
        # "remove the "c" prefix
        adata.obs[column] = adata.obs[column].str.removeprefix("c")
        # overwrite the cluster colors in adata.obs, in case a custom color palette was given
        adata.uns[f"{column}_colors"] = [
            col for c, col in cluster2color.items() if c in rename_dict[r]
        ]

    # rename the clusters in g
    node_mapping_name1 = {}
    node_mapping_name2 = {}
    node_mapping_color = {}
    for r, d in rename_dict.items():
        for c_old, c_new in d.items():
            node_old = f"{r}_{c_old}"
            node_new1 = f"{r}_{c_new}"
            node_new2 = f"{r}_{c_new[1:]}"  # remove the "c" prefix
            node_mapping_name1[node_old] = node_new1
            node_mapping_name2[node_new1] = node_new2

            color_new = cluster2color[c_new[1:]]
            node_mapping_color[node_old] = color_new

    # update the node colors
    nx.set_node_attributes(g, name="color", values=node_mapping_color)
    # update the node labels
    nx.relabel_nodes(g, node_mapping_name1, copy=False)
    nx.relabel_nodes(g, node_mapping_name2, copy=False)


def clustree_plot(
    plot_data,
    fig_scale=1.0,
    node_sizes=(100, 3000),
    edge_sizes=(1, 8),
    font_size=15,
    add_legend=True,
    subplot_kwargs=None,
    node_kwargs=None,
    edge_kwargs=None,
    label_kwargs=None,
    legend_kwargs=None,
    return_plot=False,
):
    """
    Plot the clustree.

    :param plot_data: output from clustree()
    :param fig_scale: multiplies the figure dimensions
    :param node_sizes: tuple of (min, max) node sizes
    :param edge_sizes: tuple of (min, max) edge widths
    :param font_size: label font size
    :param add_legend: if True, add a legend to the plot
    :param subplot_kwargs: kwargs passed on to plt.subplot
    :param node_kwargs: kwargs passed on to nx.draw_networkx_nodes
    :param edge_kwargs: kwargs passed on to nx.draw_networkx_edges
    :param label_kwargs: kwargs passed on to nx.draw_networkx_labels
    :param legend_kwargs: kwargs passed on to ax.legend
    :param return_plot:  if True, returns fig and ax
    :return:
    """
    g = plot_data["graph"]
    n_resolutions = plot_data["dimensions"]["n_resolutions"]
    n_clusters = plot_data["dimensions"]["n_clusters"]
    y_ticks = plot_data["axis"]["y_ticks"]
    y_labels = plot_data["axis"]["y_labels"]
    method = plot_data["cluster_method"]
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if node_kwargs is None:
        node_kwargs = {}
    if edge_kwargs is None:
        edge_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}
    if legend_kwargs is None:
        legend_kwargs = {
            "labelspacing": 2,
            "borderpad": 2,
        }

    # used to scale the sizes of the nodes and edges to the specified parameter
    node_size_max = 0
    node_size_min = float("inf")
    for n, attrs in g.nodes(data=True):
        node_size_max = max(attrs["size"], node_size_max)
        node_size_min = min(attrs["size"], node_size_min)
    edge_size_max = 0
    edge_size_min = float("inf")
    for u, v, attrs in g.edges(data=True):
        edge_size_max = max(attrs["size"], edge_size_max)
        edge_size_min = min(attrs["size"], edge_size_min)

    pos = {}
    node_sizes_ = []
    node_colors = []
    node_labels = {}
    for n, attrs in g.nodes(data=True):
        pos[n] = (attrs["x"], attrs["y"])
        # scale to [0, 1]
        s = (attrs["size"] - node_size_min) / (node_size_max - node_size_min)
        # scale to specified node_sizes
        s = (node_sizes[1] - node_sizes[0]) * s + node_sizes[0]
        node_sizes_.append(s)
        node_colors.append(attrs["color"])
        node_labels[n] = n.split("_")[1]

    edge_widths = []
    for u, v, attrs in g.edges(data=True):
        # scale to [0, 1]
        s = (attrs["size"] - edge_size_min) / (edge_size_max - edge_size_min)
        # scale to specified edge_sizes
        s = (edge_sizes[1] - edge_sizes[0]) * s + edge_sizes[0]
        edge_widths.append(s)

    fig, ax = plt.subplots(
        figsize=(fig_scale * n_resolutions, fig_scale * n_clusters), **subplot_kwargs
    )

    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes_,
        ax=ax,
        **node_kwargs,
    )

    arrows = nx.draw_networkx_edges(
        g,
        pos=pos,
        edge_color="grey",
        width=edge_widths,
        node_size=node_sizes_,
        ax=ax,
        **edge_kwargs,
    )
    # make the arrows less phallic
    for a in arrows:
        a.set_joinstyle("miter")

    nx.draw_networkx_labels(
        g,
        pos=pos,
        labels=node_labels,
        font_size=font_size,
        ax=ax,
        **label_kwargs,
    )

    if add_legend:
        # add a legend for node sizes (with max 3 elements)
        handles, labels = nodes.legend_elements(prop="sizes", alpha=0.25)
        if len(labels) > 3:
            i = len(labels) // 2
            handles = [handles[0], handles[i], handles[-1]]
            labels = [labels[0], labels[i], labels[-1]]
        ax.legend(
            handles,
            labels,
            title="Cluster sizes",
            loc="center left",
            bbox_to_anchor=(1, 0.85),
            **legend_kwargs,
        )
        fig.subplots_adjust(right=0.8)

    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax.set_yticks(ticks=y_ticks, labels=y_labels)
    ax.set_ylabel(f"{method.capitalize()} clustering resolution")
    ax.set_title("PyClustree")

    plt.show()

    if return_plot:
        return fig, ax
