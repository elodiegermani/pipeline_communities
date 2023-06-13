import numpy
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas
import seaborn
import matplotlib.pyplot as plt
import networkx as nx
from glob import glob
import numpy
from nilearn import plotting, image, masking, input_data, datasets
import warnings
import importlib
from community import community_louvain
import networkx as nx
import nibabel as nib
import numpy as np

def compute_intersection_mask(data_path, contrast):
    data_fpath = glob(f'{data_path}/*/original/n_1_*{contrast}.nii*')
    img_list = []
    mask_list = []
    target = datasets.load_mni152_gm_template(4)

    for fpath in data_fpath:
        img = nib.load(fpath)
        print('Image loaded')

        mask_img = image.binarize_img(img)
        print('Image binarized')

        resampled_mask = image.resample_to_img(
                    mask_img,
                    target,
                    interpolation='nearest')

        print('Image resampled')

        mask_list.append(resampled_mask)

    mask = masking.intersect_masks(mask_list, threshold=1)

    return mask

def compute_correlation_matrix(data_path, contrast, mask):
    target = datasets.load_mni152_gm_template(4)
    Qs=[]
    for n in range(1,1001):
        data_fpath = sorted(glob(f'{data_path}/*/original/n_{n}_contrast_{contrast}.nii*'))
        data = []
        for fpath in data_fpath:
            img = nib.load(fpath)

            resampled_gm = image.resample_to_img(
                        img,
                        target,
                       interpolation='continuous')

            masked_resampled_gm_data = resampled_gm.get_fdata() * mask.get_fdata()

            masked_resampled_gm = nib.Nifti1Image(masked_resampled_gm_data, affine=resampled_gm.affine)

            data.append(np.reshape(masked_resampled_gm_data,-1))
        Q = numpy.corrcoef(data)  
        Qs.append(Q)

    with open(f"corr_matrix_group_1000_contrast_{contrast}", "wb") as fp:   #Pickling
        pickle.dump(Qs, fp)

    return Qs

def per_group_partitioning(Qs):
    # Compute per group
    partitioning = {}
    groupnums = [i for i in range(1000)]

    for i,group in enumerate(groupnums):
        correlation_matrix = Qs[i]
        G = nx.Graph(numpy.abs(correlation_matrix))  # must be positive value for graphing
        partition = community_louvain.best_partition(G, random_state=0)
        partitioning['{}_partition'.format(group)] = [partition, G, correlation_matrix]

    return partitioning

def compute_partition_matrix(data_path, contrast, partitioning):
    data_fpath = sorted(glob(f'{data_path}/*/original/n_1_contrast_{contrast}.nii*'))
    subject = [img.split('/')[-3].split('_')[2]+','+img.split('/')[-3].split('_')[4]+','+img.split('/')[-3].split('_')[7] +',' + img.split('/')[-3].split('_')[9] for img in data_fpath]

    ##############
    # build a matrix which summarize all hypothese louvain community into one community
    ##############

    matrix_graph = numpy.zeros((24, 24))
    # teams per partition
    for key_i in partitioning.keys():
        print('\n***** Doing ****')
        print(key_i)
        # build summary matrix for alltogether matrix
        for key_j in partitioning[key_i][0].keys():
            community_key_j = partitioning[key_i][0][key_j]
            for team in range(len(partitioning[key_i][0].keys())):
                if team == key_j: # a team should not be counted as belonging to same community of itself
                    continue
                if partitioning[key_i][0][team] == community_key_j:
                    # # debugging
                    print(partitioning[key_i][0][team], " == ", community_key_j, ' thus adding 1 at row: ', subject[team], " col: ", subject[key_j])
                    matrix_graph[team][key_j] += 1

    return matrix_graph

def reorganize_with_louvain_community(matrix, partition):
    ''' Reorganized the correlation matrix according to the partition

    Parameters
    ----------
    matrix : correlation matrix (n_roi*n_roi)

    Returns
    ----------
    Dataframe reorganized as louvain community 
    '''
    # compute the best partition
    louvain = numpy.zeros(matrix.shape).astype(matrix.dtype)
    labels = range(len(matrix))
    labels_new_order = []
    
    # reorganize matrix row-wise
    i = 0
    # iterate through all created community
    for values in numpy.unique(list(partition.values())):
        # iterate through each ROI
        for key in partition:
            if partition[key] == values:
                louvain[i] = matrix[key]
                labels_new_order.append(labels[key])
                i += 1

    # checking change in positionning from original matrix to louvain matrix
    # get index of first roi linked to community 0
    index_roi_com0_louvain = list(partition.values()).index(0)
    # get nb of roi in community 0
    nb_com0 = numpy.unique(list(partition.values()), return_counts=True)[1][0]
    # # get index of first roi linked to community 1
    index_roi_com1_louvain = list(partition.values()).index(1)
    assert louvain[0].sum() == matrix[index_roi_com0_louvain].sum()
    assert louvain[nb_com0].sum() == matrix[index_roi_com1_louvain].sum() 

    df_louvain = pandas.DataFrame(index=labels_new_order, columns=labels, data=louvain)

    # reorganize matrix column-wise
    df_louvain = df_louvain[df_louvain.index]
    return df_louvain


def build_both_graph_heatmap(matrix, G, partition, title_graph, title_heatmap, subjects, hyp, saving_name):
    ''' Build and plot the graph plot next to the heatmap

    Parameters
    ----------
    matrix : correlation matrix (n_roi*n_roi)
    G: nodes and edges of the graph to plot
    partition: community affiliation
    saving_name: path to save file
    title_graph: title for the graph plot
    title_heatmap:title for the heatmap
    subjects: list of subject names (must be in same order as in the correlation matrix)
    hyp: hypothesis number being plotted

    Returns
    ----------
    plot with a graph and a heatmap
    '''
    f, axs = plt.subplots(1, 2, figsize=(30, 15)) 
    # draw the graph
    pos = nx.spring_layout(G, seed=0)
    # color the nodes according to their partition
    colors = ['blue', 'yellow', 'green', 'red', 'darkviolet', 'orange', "yellowgreen", 'lime', 'crimson', 'aqua']
    # draw edges
    nx.draw_networkx_edges(G, pos, ax=axs[0], alpha=0.06)#, min_source_margin=, min_target_margin=)
    # useful for labeling nodes
    inv_map = {k: subjects[k] for k, v in partition.items()}
    # draw nodes and labels
    for node, color in partition.items():
        nx.draw_networkx_nodes(G, pos, [node], ax=axs[0], node_size=900,
                               node_color=[colors[color]], margins=-0.01, alpha=0.35)
        # add labels to the nodes
        nx.draw_networkx_labels(G,pos,inv_map, ax=axs[0], font_size=10, font_color='black')
    axs[0].set_title(title_graph, fontsize=16)

    # add legend to the graph plot
    legend_labels = []
    for com_nb in range(max(partition.values())+1):
        patch = mpatches.Patch(color=colors[com_nb], label='Community {}'.format(com_nb))
        legend_labels.append(patch)
    axs[0].legend(handles=legend_labels, loc='lower left', handleheight=0.2)

    # draw heatmap
    matrix_organized_louvain = reorganize_with_louvain_community(matrix, partition)
    labels = [subjects[louvain_index] + "_c{}".format(partition[louvain_index]) for louvain_index in matrix_organized_louvain.columns]
    seaborn.heatmap(matrix_organized_louvain, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[1], cbar_kws={'shrink': 0.6})
    axs[1].set_title(title_heatmap, fontsize=16)
    N_team = matrix_organized_louvain.columns.__len__()
    axs[1].set_xticks(range(N_team), labels=labels, rotation=90, fontsize=7)
    axs[1].set_yticks(range(N_team), labels=labels, fontsize=7)
    plt.suptitle("Group {}".format(hyp), fontsize=20)
    plt.savefig(saving_name, dpi=300)
    plt.show()
    plt.close('all')