# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

import os
from glob import glob
from nilearn import plotting, image, masking, input_data, datasets,glm
import nibabel as nib
import seaborn
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from lib import utils
from community import community_louvain
import networkx as nx


def compute_mean_maps(data_path, contrast, mask, partition):
    data_fpath = sorted(glob(f'{data_path}/group-1_*{contrast}_*_tstat.nii*'))
    subject = [img.split('/')[-1].split('_')[2].split('-')[0].upper() +','+img.split('/')[-1].split('_')[2].split('-')[1]+','+img.split('/')[-1].split('_')[2].split('-')[2] +',' + img.split('/')[-1].split('_')[2].split('-')[3] for img in sorted(data_fpath)]
    
    target = datasets.load_mni152_gm_template(4)
    masker = input_data.NiftiMasker(
            mask_img=mask)
    if not os.path.exists(f'../figures/mean_img_community_0_con_{contrast}.nii'):
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

                nib.save(mean_img, f'../figures/mean_img_pipeline_{soft}-{f}-{p}-{h}_con_{contrast}.nii')

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
        mean_img = nib.load(f'../figures/mean_img_community_{community}_con_{contrast}.nii')
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

    fig.savefig(f'../figures/mean_maps_communities_{contrast}.png', dpi=300) 

def main():
    data_path = '/srv/tempdd/egermani/hcp_many_pipelines'
    contrast = 'left-hand'

    mask = utils.compute_intersection_mask(data_path, contrast)
    Qs = utils.compute_correlation_matrix(data_path, contrast, mask)
    partitioning = utils.per_group_partitioning(Qs)
    matrix_graph, subject = utils.compute_partition_matrix(data_path, contrast, partitioning)

    G = nx.Graph(matrix_graph, seed=0)
    # compute the best partition
    partition = community_louvain.best_partition(G, random_state=0)
    
    compute_mean_maps(data_path, contrast, mask, partition)
    plot_mean_image(contrast, partition)

if __name__ == '__main__':
	main()