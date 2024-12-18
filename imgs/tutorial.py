import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import warnings

from opticlust.clust import clustering, clustering_plot
from opticlust.tree import clustree, clustree_plot
from opticlust.recommend import score_resolutions, recommend_resolutions


# configuration
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "monospace"

# load data (this dataset has been preprocessed already)
adata = sc.datasets.pbmc68k_reduced()

# clustering & plotting (updates adata.obs)
columns = clustering(adata)
# to select a different cluster args, e.g. flavor, use:
# columns = clustering(adata, cluster_kwargs={"flavor":"leidenalg"})

# score each resolution (updates adata.uns["opticlust"])
fig, ax = score_resolutions(adata, columns, tests="SH_CH_DB", method="mean", return_plot=True)
fig.savefig("score_resolutions.png")

# to pre-select promising clustering resolutions, use:
tree_columns, fig, ax = clustering_plot(adata, columns, method="score", min_n_resolutions=1, return_plot=True)
fig.savefig("clustering_plot.png")
# otherwise, use "columns" instead of "tree_columns" in code below

# divide the resolution into 3 bins and print the best resolution per bin
top_overall, top_low, top_medium, top_high = recommend_resolutions(adata, tree_columns)

# build tree & plotting (updates adata.obs)
tree_data = clustree(adata, tree_columns, rename_cluster=False)
fig, ax = clustree_plot(tree_data, return_plot=True)
fig.savefig("clustree_plot_default.png")

# plot the UMAPs for each resolution
sc.pl.umap(adata, color=tree_columns, legend_loc="on data", alpha=0.75, ncols=3, save="/../../umaps_default.png")

# build tree & plotting (updates adata.obs)
tree_data = clustree(adata, tree_columns, rename_cluster=True)
fig, ax = clustree_plot(tree_data, return_plot=True)
fig.savefig("clustree_plot_recolored.png")

# plot the UMAPs for each resolution
sc.pl.umap(adata, color=tree_columns, legend_loc="on data", alpha=0.75, ncols=3, save="/../../umaps_recolored.png")

# plot the top genes per cluster for each resolution
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
for i, column in enumerate([top_low, top_medium, top_high]):
    sc.tl.rank_genes_groups(adata, column, n_genes=5)
    sc.tl.dendrogram(adata, column)
    sc.pl.rank_genes_groups_heatmap(adata, show_gene_labels=True, save=f"/../../top_genes_heatmap_{i}.png")
    sc.pl.rank_genes_groups_dotplot(adata, title=column, save=f"/../../top_genes_dotplot_{i}.png")
