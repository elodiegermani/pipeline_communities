import os
import glob
import numpy
from nilearn import plotting, image, masking, input_data, datasets
import warnings
import importlib
from community import community_louvain
import networkx as nx
import nibabel as nib
import numpy as np
from lib import utils

def main():
	data_path = ''
	contrast = 'rh'

	mask = utils.compute_intersection_mask(data_path, contrast)
	Qs = utils.compute_correlation_matrix(data_path, contrast, mask)
	partitioning = utils.per_group_partitioning(Qs)
	matrix_graph = utils.compute_partition_matrix(data_path, contrast, partitioning)

	G = nx.Graph(matrix_graph, seed=0)
	# compute the best partition
	partition = community_louvain.best_partition(G, random_state=0)
	#saving_name = '{}/graph_community_alltogether.png'.format(results_dir)
	title_graph = "Community of HCP many pipelines"
	title_heatmap = "Heatmap (Louvain organized) based on occurence \nof belonging to the same community across each group-level analysis"
	saving_name = f'./graph_and_heatmap_1000_groups_{contrast}.png'

	utils.build_both_graph_heatmap(matrix_graph, G, partition, title_graph, title_heatmap, subject, "All", saving_name)

if __name__ == '__main__':
	main()