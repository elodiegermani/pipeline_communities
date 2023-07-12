# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

from lib import louvain_utils, mean_maps_utils
from community import community_louvain
import networkx as nx


def main():
    # Parameters 
    data_path = '/nfs/nas-empenn/data/share/users/egermani/hcp_many_pipelines'
    repo_path = '/srv/tempdd/egermani/pipeline_distance'
    #data_path = '/Volumes/empenn/egermani/hcp_many_pipelines' # Path to data
    #repo_path = '/Users/egermani/Documents/pipeline_distance' # Path to repository (to avoid relative paths)

    contrast = 'right-foot'
    data_type='group'

    Qs = louvain_utils.compute_correlation_matrix(data_path, repo_path, contrast, data_type)
    partitioning = louvain_utils.per_group_partitioning(Qs)
    matrix_graph, subject = louvain_utils.compute_partition_matrix(data_path, partitioning, data_type)

    G = nx.Graph(matrix_graph, seed=0)
    # compute the best partition
    partition = community_louvain.best_partition(G, random_state=0)
    
    mean_maps_utils.compute_mean_maps(data_path, repo_path, contrast, partition, data_type)
    mean_maps_utils.plot_mean_image(contrast, partition, repo_path, data_type)
    mean_maps_utils.plot_pipeline_maps(contrast, partition, subject, repo_path, data_type)

if __name__ == '__main__':
	main()