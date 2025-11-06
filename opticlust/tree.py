import warnings
from itertools import cycle

import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc
from natsort import natsorted

from .utils import validate_resolutions


def clustree(
    adata,
    columns,
    rename_clusters=True,
    cluster2color=None,
    colors=None,
    node_size_min=False,
    node_multiplier=6,
    edge_multiplier=60,
    figsize=(10, 12),
    return_plot=False,
    subplot_kwargs=None,
    node_kwargs=None,
    label_kwargs=None,
    edge_kwargs=None,
    legend_kwargs=None,
):
    """
    Map the clusters to a tree structure.
    The resolution descends along the y-axis (min at the top, max at the bottom),
    and the clusters are distributed over the x-axis by size.
    Daughter clusters are plotted proximal to the parent with the highest cell overlap.

    If rename_cluster is True, clusters are updated in adata.obs and colors are updated in adata.uns
    such that cluster names and colors are consistent between resolutions.
    Note that this does **not** alter the data in any way.

    :param adata: dataset
    :param columns: a list of adata.obs column names to use.
    :param rename_clusters: change the cluster names in adata.obs
    :param cluster2color: optional dictionary to specify the cluster to color relation
    :param colors: optional list of colors (default: scanpy defaults)
    :param node_size_min: allow smaller nodes to shrink beyond label size? (default: False)
    :param node_multiplier: multiply all node sizes by this factor (default: 6)
    :param edge_multiplier: multiply all edge widths by this factor (default: 1)
    :param figsize: matplotlib figsize
    :param return_plot: if True, returns fig and ax
    :param subplot_kwargs: kwargs passed on to plt.subplot
    :param node_kwargs: kwargs passed on to nx.draw_networkx_nodes
    :param label_kwargs: kwargs passed on to nx.draw_networkx_labels
    :param edge_kwargs: kwargs passed on to nx.draw_networkx_edges
    :param legend_kwargs: kwargs passed on to ax.legend
    :return:
    """
    columns = natsorted(columns)
    method_clustering, resolutions = validate_resolutions(columns)
    if cluster2color is None:
        cluster2color = {}
        if colors is None:
            colors = sc.pl.palettes.vega_20_scanpy  # default scanpy colors
        colors = cycle(colors)
    elif colors:
        warnings.warn(
            "You provided both cluster2color and colors. "
            "Argument colors will be ignored."
        )

    # Build a cluster tree graph
    g = nx.DiGraph()
    n_cells_total = len(adata.obs)
    top_level_clusters = {}  # used to rename and recolor the clusters

    # 1) first row
    column = columns[0]
    res = column.rsplit("_", 1)[1]
    cur_clusters = []
    for node_name in adata.obs[column].cat.categories:
        node_id = f"{res}_{node_name}"

        barcodes = set(adata.obs[adata.obs[column] == node_name].index)
        cur_clusters.append(node_id)
        top_level_clusters[node_id] = len(barcodes)

        node_size = node_multiplier * len(barcodes) / n_cells_total
        if node_name not in cluster2color:
            cluster2color[node_name] = next(colors)
        g.add_node(
            node_id,
            _n_cells=len(barcodes),
            _barcodes=barcodes,
            width=node_size,
            height=node_size,
            fixedsize=not node_size_min,
            label=f"{node_name: ^3}",  # current name & color
            color=cluster2color[node_name],
            shape="circle",
            style="filled",
        )
    prev_clusters = cur_clusters

    # 2) every other row
    for column in columns[1:]:
        res = column.rsplit("_", 1)[1]
        cur_clusters = []
        for node_name in adata.obs[column].cat.categories:
            node_id = f"{res}_{node_name}"

            barcodes = set(adata.obs[adata.obs[column] == node_name].index)
            cur_clusters.append(node_id)

            # add node
            node_size = node_multiplier * len(barcodes) / n_cells_total
            if node_name not in cluster2color:
                cluster2color[node_name] = next(colors)
            g.add_node(
                node_id,
                _n_cells=len(barcodes),
                _barcodes=barcodes,
                width=node_size,
                height=node_size,
                fixedsize=not node_size_min,
                label=f"{node_name: ^3}",  # current name & color
                color=cluster2color[node_name],
                shape="circle",
                style="filled",
            )

            # add edges to parents
            for parent in prev_clusters:
                barcodes_parent = g.nodes[parent]["_barcodes"]
                n_overlap = len(barcodes & barcodes_parent)
                if n_overlap == 0:
                    continue
                # make the edge 0 width if the number of cells is too low
                # (always draw the edge in order to plot the node in the correct location)
                edge_size = edge_multiplier * n_overlap / n_cells_total
                g.add_edge(
                    parent,
                    node_id,
                    _n_overlap=n_overlap,
                    penwidth=edge_size,
                    arrowsize=0.1,  # relative to penwidth
                    color="black",
                )
        prev_clusters = cur_clusters

    if rename_clusters:
        _rename_clusters_in_graph(g, cluster2color, top_level_clusters, colors)
        _rename_clusters_in_adata(g, adata, method_clustering)

    for column in columns:
        adata.uns[f"{column}_colors"] = [
            color
            for cluster, color in cluster2color.items()
            if cluster in adata.obs[column].cat.categories
        ]

    # a = nx.drawing.nx_agraph.to_agraph(g)
    # a.layout('dot')  # untangles the edges
    # a.draw('clustree.png')
    return _plot_clustree(
        g,
        method_clustering,
        resolutions,
        figsize=figsize,
        return_plot=return_plot,
        subplot_kwargs=subplot_kwargs,
        node_kwargs=node_kwargs,
        label_kwargs=label_kwargs,
        edge_kwargs=edge_kwargs,
        legend_kwargs=legend_kwargs,
    )


def _rename_clusters_in_graph(g, cluster2color, top_level_clusters, colors):
    """
    Rename the clusters in graph g.
    Add colors to cluster2color if new clusters are added.
    """
    next_cluster_id = 0

    # Give the top level clusters a name and color in order of size
    top_level_clusters = dict(
        sorted(top_level_clusters.items(), key=lambda item: item[1], reverse=True)
    )
    for node_id in top_level_clusters:
        node_name = str(next_cluster_id)
        next_cluster_id += 1
        if node_name not in cluster2color:
            cluster2color[node_name] = next(colors)
        g.nodes[node_id]["label"] = f"{node_name: ^3}"
        g.nodes[node_id]["color"] = cluster2color[node_name]

    # Rename and recolor daughter nodes recursively.
    parents = set(top_level_clusters)
    ancestors_without_descendant = {}
    while parents:
        daughters = set()
        daughters_without_ancestor = {}
        parents_with_descendant = set()

        # 1) assign the daughter with the largest overlap a parent's name and color
        cluster2overlap = {}  # collect all outgoing edges from the parents
        for parent, daughter, md in g.out_edges(parents, data=True):
            cluster2overlap[(parent, daughter)] = md["_n_overlap"]
            if daughter not in daughters_without_ancestor:
                daughters_without_ancestor[daughter] = g.nodes[daughter]["_barcodes"]
        cluster2overlap = dict(
            sorted(cluster2overlap.items(), key=lambda item: item[1], reverse=True)
        )
        daughter_without_ancestor = dict(
            sorted(
                daughters_without_ancestor.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )
        for (parent, daughter), overlap in cluster2overlap.items():
            if daughter in daughters or parent in parents_with_descendant:
                continue  # only rename a cluster once
            g.nodes[daughter]["label"] = g.nodes[parent]["label"]
            g.nodes[daughter]["color"] = g.nodes[parent]["color"]
            parents_with_descendant.add(parent)
            del daughter_without_ancestor[daughter]
            daughters.add(daughter)

        # 2) assign the daughter with the largest overlap an ancestor's name and color
        for ancestor, barcodes_ancestor in list(ancestors_without_descendant.items()):
            best = None, 0
            for daughter, barcodes in daughter_without_ancestor.items():
                n_overlap = len(barcodes & barcodes_ancestor)
                if n_overlap > best[1]:
                    best = daughter, n_overlap
            if best[1] > 0:
                daughter = best[0]
                g.nodes[daughter]["label"] = g.nodes[ancestor]["label"]
                g.nodes[daughter]["color"] = g.nodes[ancestor]["color"]
                del ancestors_without_descendant[ancestor]
                del daughter_without_ancestor[daughter]
                daughters.add(daughter)

        # 3) assign the daughter a new name and color
        for daughter in list(daughter_without_ancestor):
            node_name = str(next_cluster_id)
            next_cluster_id += 1
            if node_name not in cluster2color:
                cluster2color[node_name] = next(colors)
            g.nodes[daughter]["label"] = f"{node_name: ^3}"
            g.nodes[daughter]["color"] = cluster2color[node_name]
            daughters.add(daughter)

        # update ancestors_without_descendant
        for parent in parents - parents_with_descendant:
            ancestors_without_descendant[parent] = g.nodes[parent]["_barcodes"]
        ancestors_without_descendant = dict(
            sorted(
                ancestors_without_descendant.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )

        # next iteration
        parents = daughters


def _rename_clusters_in_adata(g, adata, method_clustering):
    to_rename = {}
    for node_id, md in g.nodes(data=True):
        res, cluster = node_id.split("_", 1)
        if res not in to_rename:
            to_rename[res] = {}
        to_rename[res][cluster] = md["label"].strip()
    for res, d in to_rename.items():
        column = f"{method_clustering}_res_{float(res):4.2f}"
        adata.obs[column] = (
            adata.obs[column]
            .cat.rename_categories(d)
            .cat.reorder_categories(natsorted(d.values()))
        )


def _plot_clustree(
    g,
    method_clustering,
    resolutions,
    figsize=None,
    return_plot=False,
    subplot_kwargs=None,
    node_kwargs=None,
    label_kwargs=None,
    edge_kwargs=None,
    legend_kwargs=None,
):
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if node_kwargs is None:
        node_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}
    if edge_kwargs is None:
        edge_kwargs = {
            "arrows": True,
            "edge_color": "black",
            "arrowstyle": "-|>",  # I personally think this style scales better
        }
    if legend_kwargs is None:
        legend_kwargs = {
            # "labelspacing": 2,
            # "borderpad": 2,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Warning: node [...] size too small for label
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")  # untangles the edges

    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        node_color=nx.get_node_attributes(g, "color").values(),  # noqa
        node_size=[300 * n for n in nx.get_node_attributes(g, "height").values()],
        ax=ax,
        **node_kwargs,
    )
    nx.draw_networkx_labels(
        g,
        pos=pos,
        labels=nx.get_node_attributes(g, "label"),
        ax=ax,
        **label_kwargs,
    )
    edge_widths = [0.3 * n for n in nx.get_edge_attributes(g, "penwidth").values()]
    arrows = nx.draw_networkx_edges(
        g, pos=pos, width=edge_widths, ax=ax, **edge_kwargs  # noqa
    )
    for a in arrows:
        a.set_joinstyle("miter")  # makes arrows less phallic

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

    # add a legend for edge widths
    # using a custom legend because the edges are too large in the legend
    e = sorted(set(edge_widths))
    e_mn = min(edge_widths)
    e_q1 = e[1 * len(e) // 4]
    e_q2 = e[2 * len(e) // 4]
    e_q3 = e[3 * len(e) // 4]
    e_mx = max(edge_widths)
    overlaps = list(nx.get_edge_attributes(g, "_n_overlap").values())
    (e_mn_h,) = plt.plot(
        [], [], color="black", linewidth=e_mn, label=overlaps[edge_widths.index(e_mn)]
    )
    (e_q1_h,) = plt.plot(
        [], [], color="black", linewidth=e_q1, label=overlaps[edge_widths.index(e_q1)]
    )
    (e_q2_h,) = plt.plot(
        [], [], color="black", linewidth=e_q2, label=overlaps[edge_widths.index(e_q2)]
    )
    (e_q3_h,) = plt.plot(
        [], [], color="black", linewidth=e_q3, label=overlaps[edge_widths.index(e_q3)]
    )
    (e_mx_h,) = plt.plot(
        [], [], color="black", linewidth=e_mx, label=overlaps[edge_widths.index(e_mx)]
    )
    leg2 = ax.legend(
        handles=[e_mn_h, e_q1_h, e_q2_h, e_q3_h, e_mx_h],
        title="Cluster overlap",
        loc="center left",
        bbox_to_anchor=(1, 0.50),
        **legend_kwargs,
    )
    ax.add_artist(leg2)

    ax.set_title("opticlust")
    ax.set_ylabel(f"{method_clustering.capitalize()} cluster resolution")
    ax.set_yticks(sorted({p[1] for p in pos.values()}, reverse=True))
    ax.set_yticklabels(resolutions)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=True, labelleft=True)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)

    if return_plot:
        return fig, ax
    else:
        plt.show()
