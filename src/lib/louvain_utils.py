import numpy
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import pandas
import seaborn
import matplotlib.pyplot as plt
import networkx as nx
from glob import glob
import numpy
from nilearn import plotting, image, masking, input_data, datasets, glm
import warnings
import importlib
from community import community_louvain
import networkx as nx
import nibabel as nib
import numpy as np
import os
import pickle

def compute_intersection_mask(data_path, contrast):
    data_fpath = glob(f'{data_path}/group-1_*{contrast}_*_tstat.nii*')
    img_list = []
    mask_list = []
    target = datasets.load_mni152_gm_template(4)
    print('Computing mask...')
    for fpath in data_fpath:
        img = nib.load(fpath)

        mask_img = image.binarize_img(img)

        resampled_mask = image.resample_to_img(
                    mask_img,
                    target,
                    interpolation='nearest')

        mask_list.append(resampled_mask)

    mask = masking.intersect_masks(mask_list, threshold=1)

    return mask

def compute_correlation_matrix(data_path, repo_path, contrast):
    target = datasets.load_mni152_gm_template(4)

    if not os.path.exists(f"{repo_path}/figures/corr_matrix_1000_groups_{contrast}"):
        print('Computing correlation matrix...')
        mask = compute_intersection_mask(data_path, contrast)
        Qs=[]
        for n in range(1,1001):
            data_fpath = sorted(glob(f'{data_path}/group-{n}_{contrast}_*_tstat.nii'))
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
            
        with open(f"{repo_path}/figures/corr_matrix_1000_groups_{contrast}", "wb") as fp:   #Pickling
            pickle.dump(Qs, fp)

    else:
        with open(f"{repo_path}/figures/corr_matrix_1000_groups_{contrast}", "rb") as fp:   #Pickling
            Qs=pickle.load(fp)

    return Qs

def per_group_partitioning(Qs):
    # Compute per group
    partitioning = {}
    groupnums = [i for i in range(len(Qs))]

    for i,group in enumerate(groupnums):
        correlation_matrix = Qs[i]
        G = nx.Graph(numpy.abs(correlation_matrix))  # must be positive value for graphing
        partition = community_louvain.best_partition(G, random_state=0)
        partitioning['{}_partition'.format(group)] = [partition, G, correlation_matrix]

    return partitioning

def compute_partition_matrix(data_path, contrast, partitioning):
    data_fpath = sorted(glob(f'{data_path}/group-1_*{contrast}_*_tstat.nii*'))
    subject = [img.split('/')[-1].split('_')[2].split('-')[0].upper() +','+img.split('/')[-1].split('_')[2].split('-')[1]+','+img.split('/')[-1].split('_')[2].split('-')[2] +',' + img.split('/')[-1].split('_')[2].split('-')[3] for img in sorted(data_fpath)]

    ##############
    # build a matrix which summarize all hypothese louvain community into one community
    ##############

    matrix_graph = numpy.zeros((24, 24))
    # teams per partition
    for key_i in partitioning.keys():
        #print('\n***** Doing ****')
        #print(key_i)
        # build summary matrix for alltogether matrix
        for key_j in partitioning[key_i][0].keys():
            community_key_j = partitioning[key_i][0][key_j]
            for team in range(len(partitioning[key_i][0].keys())):
                if team == key_j: # a team should not be counted as belonging to same community of itself
                    continue
                if partitioning[key_i][0][team] == community_key_j:
                    # # debugging
                    #print(partitioning[key_i][0][team], " == ", community_key_j, ' thus adding 1 at row: ', subject[team], " col: ", subject[key_j])
                    matrix_graph[team][key_j] += 1

    return matrix_graph, subject

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


def build_both_graph_heatmap(matrix, G, partition, subjects, hyp, saving_names, contrast):
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
    f, axs = plt.subplots(1, 1, figsize=(20, 20)) 
    f.suptitle(f'{contrast.upper()}', size=28, fontweight='bold', backgroundcolor= 'black', color='white')
    # draw the graph
    pos = nx.spring_layout(G, seed=0)
    # color the nodes according to their partition
    colors = ['blue', 'orange', 'green', 'red', 'darkviolet', 'yellow', "yellowgreen", 'lime', 'crimson', 'aqua']
    # draw edges
    nx.draw_networkx_edges(G, pos, ax=axs, alpha=0.06)#, min_source_margin=, min_target_margin=)
    # useful for labeling nodes
    inv_map = {k: subjects[k] for k, v in partition.items()}
    # draw nodes and labels
    for node, color in partition.items():
        nx.draw_networkx_nodes(G, pos, [node], ax=axs, node_size=900,
                               node_color=[colors[color]], margins=-0.01, alpha=0.35)
        # add labels to the nodes
        nx.draw_networkx_labels(G,pos,inv_map, ax=axs, font_size=20, font_color='black')
    #axs[0].set_title(title_graph, fontsize=16)

    # add legend to the graph plot
    legend_labels = []
    for com_nb in range(max(partition.values())+1):
        patch = mpatches.Patch(color=colors[com_nb], label='Community {}'.format(com_nb+1))
        legend_labels.append(patch)
    axs.legend(handles=legend_labels, loc='lower left', handleheight=0.2)
    
    plt.savefig(saving_names[0], dpi=300)
    plt.show()
    plt.close()

    f, axs = plt.subplots(1, 1, figsize=(20, 20)) 
    if contrast == 'all':
        max_val = 2500
        vmax=5000
    else:
        max_val = 500
        vmax=1000
    # draw heatmap
    matrix_organized_louvain = reorganize_with_louvain_community(matrix, partition)
    labs = np.array(matrix_organized_louvain).astype('int').astype('str')
    labels = [subjects[louvain_index] + "_c{}".format(partition[louvain_index]+1) for louvain_index in matrix_organized_louvain.columns]
    cm = seaborn.heatmap(matrix_organized_louvain, mask = matrix_organized_louvain > max_val, center=0, cmap='coolwarm', robust=True, square=True, ax=axs, cbar_kws={'shrink': 0.6}, xticklabels=False, annot=labs, fmt='', cbar=False, vmin=0, vmax=vmax)
    cm = seaborn.heatmap(matrix_organized_louvain, mask = matrix_organized_louvain < max_val, center=0, cmap='coolwarm', robust=True, square=True, ax=axs, cbar_kws={'shrink': 0.6}, xticklabels=False, annot=labs, fmt='', annot_kws={"weight": "bold", "size":13}, vmin=0, vmax=vmax)
    N_team = matrix_organized_louvain.columns.__len__()
    axs.set_xticks(range(N_team), labels=labels, rotation=90, fontsize=22)
    axs.set_yticks(range(N_team), labels=labels, rotation=360, fontsize=22)

    for i, ticklabel in enumerate(cm.axes.yaxis.get_majorticklabels()):
        color_tick = colors[int(ticklabel.get_text()[-1])-1]
        ticklabel.set_color(color_tick)

    for i, ticklabel in enumerate(cm.axes.xaxis.get_majorticklabels()):
        color_tick = colors[int(ticklabel.get_text()[-1])-1]
        ticklabel.set_color(color_tick)

    
    #plt.suptitle("Group {}".format(hyp), fontsize=20)
    plt.savefig(saving_names[1], dpi=300)
    plt.show()
    plt.close()

def compute_mean_maps(data_path, contrast, mask, partition):
    data_fpath = sorted(glob(f'{data_path}/group-1_*{contrast}_*_tstat.nii*'))
    subject = [img.split('/')[-1].split('_')[2].split('-')[0].upper() +','+img.split('/')[-1].split('_')[2].split('-')[1]+','+img.split('/')[-1].split('_')[2].split('-')[2] +',' + img.split('/')[-1].split('_')[2].split('-')[3] for img in sorted(data_fpath)]
    
    target = datasets.load_mni152_gm_template(4)
    masker = input_data.NiftiMasker(
            mask_img=mask)
    if not os.path.exists(f'../../figures/mean_img_community_0_con_{contrast}.nii'):
        print('Computing mean image...')
        for community in np.unique(list(partition.values())):
            pipelines = [sub for i, sub in enumerate(subject) if partition[i]==community]
            print('Pipelines belonging to', community, ':', pipelines)
            mean_data=[]
            for pi in pipelines:
                data = []
                soft = pi.split(',')[0].lower()
                f = pi.split(',')[1]
                p = pi.split(',')[2]
                h = pi.split(',')[3]

                data_fpath = sorted(glob(f'{data_path}/group-*_{contrast}_{soft}-{f}-{p}-{h}_tstat.nii'))

                for fpath in data_fpath:
                    resampled_gm = image.resample_to_img(
                                nib.load(fpath),
                                target,
                               interpolation='continuous')

                    data.append(resampled_gm)

                maskdata = masker.fit_transform(data)
                meandata = np.mean(maskdata, 0)
                mean_img = masker.inverse_transform(meandata)

                nib.save(mean_img, f'../../figures/mean_img_pipeline_{soft}-{f}-{p}-{h}_con_{contrast}.nii')

                mean_data.append(mean_img)

            mask_mean_data = masker.fit_transform(mean_data)
            meandata = np.mean(mask_mean_data, 0)
            mean_img = masker.inverse_transform(meandata)

            nib.save(mean_img, f'../figures/mean_img_community_{community}_con_{contrast}.nii')
    else:
        print('Mean image already computed.')
        
def plot_mean_image(contrast, partition):
    n_communities=len(np.unique(list(partition.values())))
    fig = plt.figure(figsize = (7 * n_communities, 14))
    gs = fig.add_gridspec(2, n_communities)
    
    fig.suptitle(f'{contrast.upper()}', size=28, fontweight='bold', backgroundcolor= 'black', color='white')
    for community in range(n_communities):
        mean_img = nib.load(f'../../figures/mean_img_community_{community}_con_{contrast}.nii')
        ax = fig.add_subplot(gs[0, int(community)])

        disp = plotting.plot_glass_brain(mean_img, display_mode = 'z', colorbar = True, annotate=False, 
                                                 cmap=nilearn_cmaps['cold_hot'], plot_abs=False, figure=fig, axes=ax)
        disp.title(f'Community {community+1}', size=28, fontweight='bold')

        thresh_mean_img, threshold = glm.threshold_stats_img(mean_img, alpha=0.05, height_control='fdr',
                                                         two_sided=False)
        ax = fig.add_subplot(gs[1, int(community)])

        disp2 = plotting.plot_glass_brain(thresh_mean_img, display_mode = 'z', colorbar = True, annotate=False, 
                                                 cmap=nilearn_cmaps['cold_hot'], plot_abs=False, figure=fig, axes=ax)
        disp2.title(f'Community {community+1}', size=28, fontweight='bold')

    fig.savefig(f'../../figures/mean_maps_communities_{contrast}.png', dpi=300) 
