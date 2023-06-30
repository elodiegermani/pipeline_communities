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
from lib import louvain_utils
import itertools

def main():
	# Parameters 
	#data_path = '/nfs/nas-empenn/data/share/users/egermani/hcp_many_pipelines'
	#repo_path = '/srv/tempdd/egermani/pipeline_distance'
	data_path = '/Volumes/empenn/egermani/hcp_many_pipelines' # Path to data
	repo_path = '/Users/egermani/Documents/pipeline_distance' # Path to repository (to avoid relative paths)

	contrast = 'left-hand'
	data_type='sub'


	if contrast != 'all':
		Qs = louvain_utils.compute_correlation_matrix(data_path, repo_path, contrast, data_type)

	else: # Concat all matrix 
		contrast_list = ['right-hand', 'right-foot', 'left-hand', 'left-foot', 'tongue']
		Qs_unchain = []

		for con in contrast_list: 
			Qs_unchain.append(louvain_utils.compute_correlation_matrix(data_path, repo_path, con, data_type))

		Qs = list(itertools.chain(*Qs_unchain))

	# Partition each group/subject level correlation matrix
	partitioning = louvain_utils.per_group_partitioning(Qs)

	# Compute matrix of belonging to the same community across groups/subject for each pair of pipeline
	matrix_graph, subject = louvain_utils.compute_partition_matrix(data_path, partitioning, data_type)

	G = nx.Graph(matrix_graph, seed=0)
	# Compute communities depending on the number of times two pipelines belong to the same community across groups/subjects
	partition = community_louvain.best_partition(G, random_state=0)

	saving_names = [f'{repo_path}/figures/graph_{len(Qs)}_{data_type}s_{contrast}.png',
	f'{repo_path}/figures/heatmap_{len(Qs)}_{data_type}s_{contrast}.png']

	# Plot results 
	louvain_utils.build_both_graph_heatmap(matrix_graph, G, partition, subject, saving_names, contrast, data_type)
		
if __name__ == '__main__':
	main()
