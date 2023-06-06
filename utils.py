import numpy
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas
import seaborn
import matplotlib.pyplot as plt
import networkx as nx

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


def build_both_graph_heatmap(matrix, G, partition, title_graph, title_heatmap, subjects, hyp):
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
    #plt.savefig(saving_name, dpi=300)
    plt.show()
    plt.close('all')