import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from opticlust.clust import clustering, clustering_plot
from opticlust.recommend import recommend_resolutions, score_resolutions
from opticlust.tree import clustree, clustree_plot

# configuration
matplotlib.use("TkAgg")
plt.rcParams["font.family"] = "monospace"

# load data (this dataset has been preprocessed already)
adata = sc.datasets.pbmc68k_reduced()

# clustering & plotting (updates adata.obs)
columns = clustering(adata)
# to select a different cluster args, e.g. flavor, use:
# columns = clustering(adata, cluster_kwargs={"flavor":"leidenalg"})

# score each resolution (updates adata.uns["opticlust"])
score_resolutions(adata, columns, tests="SH_CH_DB", method="mean")

# to pre-select promising clustering resolutions, use:
tree_columns = clustering_plot(adata, columns, method="score", min_n_resolutions=1)
# otherwise, use "columns" instead of "tree_columns" in code below

# divide the resolution into 3 bins and print the best resolution per bin
top_overall, top_low, top_medium, top_high = recommend_resolutions(adata, tree_columns)

# build tree & plotting (updates adata.obs)
tree_data = clustree(adata, tree_columns, rename_cluster=True)
clustree_plot(tree_data)

# plot the UMAPs for each resolution
sc.pl.umap(adata, color=tree_columns, legend_loc="on data", alpha=0.75, ncols=3)

# plot the top genes per cluster for each resolution
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
for column in [top_low, top_medium, top_high]:
    sc.tl.rank_genes_groups(adata, column, n_genes=5)
    sc.tl.dendrogram(adata, column)
    sc.pl.rank_genes_groups_heatmap(adata, show_gene_labels=True)
    sc.pl.rank_genes_groups_dotplot(adata, title=column)
