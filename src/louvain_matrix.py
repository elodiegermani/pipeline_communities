# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

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
from lib import louvain_utils, louvain_subject
import itertools

def main():
	data_path = '/nfs/nas-empenn/data/share/users/egermani/hcp_many_pipelines'
	repo_path = '/srv/tempdd/egermani/pipeline_distance'
	contrast = 'all'

	data_type='group'

	if data_type =='group':
		if contrast != 'all':
			Qs = louvain_utils.compute_correlation_matrix(data_path, repo_path, contrast)

		else:
			contrast_list = ['right-hand', 'right-foot', 'left-hand', 'left-foot', 'tongue']
			Qs_unchain = []

			for con in contrast_list: 
				Qs_unchain.append(louvain_utils.compute_correlation_matrix(data_path, repo_path, con))

			Qs = list(itertools.chain(*Qs_unchain))

		partitioning = louvain_utils.per_group_partitioning(Qs)
		matrix_graph, subject = louvain_utils.compute_partition_matrix(data_path, 'right-hand', partitioning)

		G = nx.Graph(matrix_graph, seed=0)
		# compute the best partition
		partition = community_louvain.best_partition(G, random_state=0)

		saving_names = [f'{repo_path}/figures/graph_1000_groups_{contrast}.png',
		f'{repo_path}/figures/heatmap_1000_groups_{contrast}.png']

		louvain_utils.build_both_graph_heatmap(matrix_graph, G, partition, subject, "All", saving_names, contrast)

	else: 
		mask = louvain_subject.compute_intersection_mask(data_path, contrast)
		Qs = louvain_subject.compute_correlation_matrix(data_path, contrast, mask)
		partitioning = louvain_subject.per_group_partitioning(Qs)
		matrix_graph, subject = louvain_subject.compute_partition_matrix(data_path, contrast, partitioning)

		G = nx.Graph(matrix_graph, seed=0)
		# compute the best partition
		partition = community_louvain.best_partition(G, random_state=0)

		saving_names = [f'{repo_path}/figures/graph_1080_subs_{contrast}.png',
		f'{repo_path}/figures/heatmap_1080_subs_{contrast}.png']

		louvain_subject.build_both_graph_heatmap(matrix_graph, G, partition, subject, "All", saving_names, contrast)
		
if __name__ == '__main__':
	main()
