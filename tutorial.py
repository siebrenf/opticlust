import matplotlib.pyplot as plt
import scanpy as sc

from pyclustree.clust import clustering, clustering_plot
from pyclustree.tree import clustree, clustree_plot

# configuration
plt.rcParams["font.family"] = "monospace"

# load data (this dataset has been preprocessed already)
adata = sc.datasets.pbmc68k_reduced()

# clustering & plotting (updates adata.obs)
columns = clustering(adata)
tree_columns = clustering_plot(adata, columns)

# build tree & plotting (updates adata.obs)
tree_data = clustree(adata, tree_columns, rename_cluster=True)
clustree_plot(tree_data)

# plot the UMAPs for each resolution
sc.pl.umap(adata, color=tree_columns, legend_loc="on data", alpha=0.75, ncols=3)

# plot the top genes per cluster for each resolution
for column in tree_columns:
    sc.tl.rank_genes_groups(adata, column)
    sc.tl.dendrogram(adata, column)
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, show_gene_labels=True)
