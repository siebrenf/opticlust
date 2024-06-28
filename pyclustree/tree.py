import warnings

import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc
from natsort import natsorted


def clustree(adata, columns, rename_cluster=True, cluster2color=None, colors=None):
    """
    Map the clusters to a tree structure.
    The resolution descends along the y-axis (min at the top, max at the bottom),
    and the clusters are distributed over the x-axis by size.
    Daughter clusters are plotted proximal to the parent with the highest cell overlap.

    If rename_cluster is True, clusters are updated in adata.obs and colors are updated in adata.uns
    such that the color is passed from parent the daughter with the largest cell overlap.
    Note that this does **not** alter the clusters or data in any way.

    :param adata: dataset
    :param columns: a list of adata.obs column names to use.
    :param rename_cluster: change the cluster names in adata.obs
    :param cluster2color: optional dictionary to specify the cluster to color relation
    :param colors: optional list of colors (default: scanpy defaults)
    :return: plot_data: a dictionary used in clustree_plot
    """
    columns = natsorted(columns)
    n_clusters = 0
    n_resolutions = len(columns)
    column = None  # column with the highest number of clusters
    for c in columns:
        if c not in adata.obs:
            raise ValueError(f"columns not found in adata.obs: '{c}'")
        n = len(adata.obs[c].unique())
        if n > n_clusters:
            n_clusters = n
            column = c
    n_cells = len(adata.obs)

    g = nx.DiGraph()
    y_ticks = []  # plot label
    y_labels = []  # plot label
    if column.count("_") != 2:
        raise ValueError("Column names must be in the shape '[method]_res_[res]'")
    method = column.split("_", 1)[0]
    if method not in ["leiden", "louvain"]:
        raise ValueError(
            "Column names must be in the shape '[method]_res_[res]' (with [method] leiden or louvain)"
        )
    try:
        float(column.rsplit("_", 1)[1])
    except ValueError:
        raise ValueError(
            "Column names must be in the shape '[method]_res_[res]' (with [res] a float, e.g. 0.53)"
        )

    # set cluster colors
    if cluster2color is None:
        cluster2color = {}
        if colors is None:
            colors = sc.pl.palettes.vega_20_scanpy  # default scanpy colors
        i = 0
        for c in natsorted(adata.obs[column].unique()):
            # repeat colors if there are more clusters than colors
            if i >= len(colors):
                i -= len(colors)
            cluster2color[c] = colors[i]
            i += 1
    elif colors:
        warnings.warn(
            "You provided both cluster2color and colors. "
            "Argument colors will be ignored."
        )
    # (over)write the cluster colors in adata.obs, so they will match the tree plot colors
    for column in columns:
        adata.uns[f"{column}_colors"] = [
            color
            for cluster, color in cluster2color.items()
            if cluster in adata.obs[column].unique()
        ]

    cluster2barcodes = {}
    r_prev = None
    y = n_resolutions - 1  # plot coordinate
    for column in columns:
        r = column.rsplit("_", 1)[1]  # r = resolution, e.g. "0.30"
        assert column == f"{method}_res_{r}"
        y_ticks.append(y)
        y_labels.append(r)
        cluster2barcodes[r] = {}
        for c in natsorted(adata.obs[column].unique()):  # c = cluster name, e.g. "3"
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
                g.add_node(name, x=x_node, y=y_node, size=size_node, label=c)

                x = x + dx

        else:

            # assign each cluster to the parent with the largest overlap
            edges = {}
            for c in cluster2barcodes[r]:
                best = -1, []
                name = f"{r}_{c}"
                for c_prev, barcodes_prev in cluster2barcodes[r_prev].items():
                    name_prev = f"{r_prev}_{c_prev}"
                    if name_prev not in edges:
                        edges[name_prev] = {}

                    barcodes = cluster2barcodes[r][c]
                    overlap = len(barcodes & barcodes_prev)
                    if overlap > best[0]:
                        best = overlap, [name_prev]
                    elif overlap == best[0]:  # edge case
                        best[1].append(name_prev)

                    # add the edge to the graph
                    if overlap:
                        g.add_edge(name_prev, name, size=overlap)

                overlap, names_prev = best
                for name_prev in names_prev:
                    edges[name_prev][name] = overlap

            # per parent cluster, sort daughter clusters by overlap (descending)
            for name_prev, md in edges.items():
                md = dict(sorted(md.items(), key=lambda item: item[1], reverse=True))
                edges[name_prev] = md

            # add the daughters to the graph
            cluster2barcodes_reordered = {}
            for name_prev, md in edges.items():
                for name, overlap in md.items():

                    # mirror sorting in cluster2barcodes
                    c = name.split("_")[1]
                    barcodes = cluster2barcodes[r][c]
                    cluster2barcodes_reordered[c] = barcodes

                    size_node = len(barcodes)
                    dx = (size_node / n_cells) * n_clusters
                    x_node = x + dx / 2
                    y_node = y
                    g.add_node(name, x=x_node, y=y_node, size=size_node, label=c)

                    x = x + dx

            cluster2barcodes[r] = cluster2barcodes_reordered

        y -= 1
        r_prev = r

    if rename_cluster:
        _clustering_rename(adata, g, cluster2barcodes, method)

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
        "cluster_colors": cluster2color,
    }
    return plot_data


def _clustering_rename(adata, g, cluster2barcodes, method):
    """
    Rename the clusters names, such that the clusters names remain
    consistent between resolutions.

    Updates cluster IDs in adata.obs and g.nodes(labels).
    """
    rename_dict = {}

    # bottom row: 1, 2, 3, 4, etc.
    r = list(cluster2barcodes)[-1]
    rename_dict[r] = {}
    for n, c in enumerate(cluster2barcodes[r]):
        rename_dict[r][c] = f"c{n}"
    r_daughters = r

    # other rows: assign cluster to the largest daughter
    for r_parents in list(cluster2barcodes)[:-1][::-1]:
        rename_dict[r_parents] = {}
        seen = []
        buds = []
        for c_parent, barcodes_parent in cluster2barcodes[r_parents].items():
            best = None, 0
            for c_daughter, barcodes_daughter in cluster2barcodes[r_daughters].items():
                if c_daughter in seen:
                    continue

                overlap = len(barcodes_daughter & barcodes_parent)
                if overlap > best[1]:
                    best = c_daughter, overlap

            c_daughter = best[0]
            if c_daughter is None:
                buds.append(c_parent)
                continue
            rename_dict[r_parents][c_parent] = rename_dict[r_daughters][c_daughter]
            seen.append(c_daughter)

        # when parent clusters merge into one daughter,
        # rename the smaller parent with leftover cluster names.
        if buds:
            clusters = natsorted(set(rename_dict[r_daughters]) - set(seen))
            for i, c_parent in enumerate(buds):
                rename_dict[r_parents][c_parent] = rename_dict[r_daughters][clusters[i]]

        r_daughters = r_parents

    # re-order the names so they ascend from top to bottom
    rename_dict2 = {}
    clusters = set()
    for r in rename_dict:
        for c in rename_dict[r]:
            clusters.add(rename_dict[r][c])
    clusters = natsorted(clusters)
    c2c = {}
    for r in cluster2barcodes:
        rename_dict2[r] = {}
        for c_old in cluster2barcodes[r]:
            c_new = rename_dict[r][c_old]
            if c_new not in c2c:
                c2c[c_new] = clusters.pop(0)
            rename_dict2[r][c_old] = c2c[c_new]
    rename_dict = rename_dict2

    # rename the clusters in adata
    for r, d in rename_dict.items():
        column = f"{method}_res_{r}"
        # adata.obs[column] = (
        #     adata.obs[column]
        #         .astype("str")  # convert to string
        #         .rename(d)  # relabel the clusters with "c" prefix
        #         .str.removeprefix("c")  # "remove the "c" prefix
        #         # .astype("category")  # convert back to categorical
        # )
        adata.obs[column] = adata.obs[column].cat.rename_categories(d)
        # "remove the "c" prefix
        adata.obs[column] = adata.obs[column].str.removeprefix("c")

    # leave the node name, but rename the node label
    for node, md in g.nodes(data=True):
        r, c = node.split("_")
        label = rename_dict[r][c]
        md["label"] = label[1:]  # "remove the "c" prefix


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
    cluster2color = plot_data["cluster_colors"]
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
        s = 1
        if node_size_max != node_size_min:
            s = (attrs["size"] - node_size_min) / (node_size_max - node_size_min)
        # scale to specified node_sizes
        s = (node_sizes[1] - node_sizes[0]) * s + node_sizes[0]
        node_sizes_.append(s)
        node_colors.append(cluster2color[attrs["label"]])
        node_labels[n] = attrs["label"]

    edge_widths = []
    for u, v, attrs in g.edges(data=True):
        # scale to [0, 1]
        s = 1
        if edge_size_max != edge_size_min:
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
        leg1 = ax.legend(
            handles,
            labels,
            title="Cluster sizes",
            loc="center left",
            bbox_to_anchor=(1, 0.85),
            **legend_kwargs,
        )
        ax.add_artist(leg1)

        overlaps = sorted(nx.get_edge_attributes(g, "size").values())
        handles = arrows
        labels = overlaps
        if len(handles) > 8:
            ews = sorted(edge_widths)
            i = len(edge_widths) // 8
            handles = [
                arrows[edge_widths.index(ews[i * 1])],
                arrows[edge_widths.index(ews[i * 4])],
                arrows[edge_widths.index(ews[i * 7])],
            ]
            i = len(overlaps) // 8
            labels = [
                overlaps[i * 1],
                overlaps[i * 4],
                overlaps[i * 7],
            ]
        leg2 = ax.legend(
            handles,
            labels,
            title="Cluster overlap",
            loc="center left",
            bbox_to_anchor=(1, 0.50),
            **legend_kwargs,
        )
        ax.add_artist(leg2)

        fig.subplots_adjust(right=0.8)

    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax.set_yticks(ticks=y_ticks, labels=y_labels)
    ax.set_ylabel(f"{method.capitalize()} clustering resolution")
    ax.set_title("PyClustree")

    plt.show()

    if return_plot:
        return fig, ax
