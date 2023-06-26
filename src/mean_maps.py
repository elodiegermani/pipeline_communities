# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

from lib import louvain_utils
from community import community_louvain
import networkx as nx


def main():
    data_path = '/srv/tempdd/egermani/hcp_many_pipelines'
    contrast = 'left-hand'

    mask = louvain_utils.compute_intersection_mask(data_path, contrast)
    Qs = louvain_utils.compute_correlation_matrix(data_path, contrast, mask)
    partitioning = louvain_utils.per_group_partitioning(Qs)
    matrix_graph, subject = louvain_utils.compute_partition_matrix(data_path, contrast, partitioning)

    G = nx.Graph(matrix_graph, seed=0)
    # compute the best partition
    partition = community_louvain.best_partition(G, random_state=0)
    
    louvain_utils.compute_mean_maps(data_path, contrast, mask, partition)
    louvain_utils.plot_mean_image(contrast, partition)

if __name__ == '__main__':
	main()