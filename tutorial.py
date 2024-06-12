import matplotlib.pyplot as plt
import scanpy as sc

from pyclustree.clust import clustering, clustering_plot
from pyclustree.tree import clustree, clustree_plot

# configuration
plt.rcParams["font.family"] = "monospace"

# load data (this dataset has been preprocessed already)
adata = sc.datasets.pbmc68k_reduced()

# create a UMAP
sc.tl.umap(adata)

# clustering (updates adata.obs)
columns = clustering(adata)
cluster_resolutions = clustering_plot(adata, columns)

# plotting (updates adata.obs)
plot_data = clustree(adata, cluster_resolutions, rename_cluster=False)
clustree_plot(plot_data)

# plot the UMAPs for each resolution
plot_columns = [c for n, c in cluster_resolutions.items() if n > 1]  # len(clusters)>1
sc.pl.umap(adata, color=plot_columns, legend_loc="on data", alpha=0.75, ncols=3)

# plot the top genes per cluster for each resolution
for column in plot_columns:
    sc.tl.rank_genes_groups(adata, column)
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, show_gene_labels=True)
