from lib import louvain_utils
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import matplotlib.pyplot as plt
from glob import glob
from nilearn import plotting, image, masking, input_data, datasets, glm
import nibabel as nib
import numpy as np
import os

def compute_mean_maps(data_path, repo_path, contrast, partition, data_type):
    if not os.path.exists(f'{repo_path}/figures/mean_img_{data_type}_community_0_con_{contrast}.nii'):
        if data_type == 'group':
            data_fpath = f'{data_path}/group-1_{contrast}_*_tstat.nii*'

        elif data_type == 'sub':
            data_fpath = f'{data_path}/sub-100206_{contrast}_*_tstat.nii*'
        
        subject = [img.split('/')[-1].split('_')[2].split('-')[0].upper() +','+img.split('/')[-1].split('_')[2].split('-')[1]+','+img.split('/')[-1].split('_')[2].split('-')[2] +',' + img.split('/')[-1].split('_')[2].split('-')[3] for img in sorted(glob(data_fpath))]
        
        target = datasets.load_mni152_gm_template(4)
        mask = louvain_utils.compute_intersection_mask(data_fpath)
        masker = input_data.NiftiMasker(
                mask_img=mask)

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

                data_fpath = sorted(glob(f'{data_path}/{data_type}-*_{contrast}_{soft}-{f}-{p}-{h}_tstat.nii'))

                for fpath in data_fpath:
                    resampled_gm = image.resample_to_img(
                                nib.load(fpath),
                                target,
                               interpolation='continuous')

                    data.append(resampled_gm)

                maskdata = masker.fit_transform(data)
                meandata = np.mean(maskdata, 0)
                mean_img = masker.inverse_transform(meandata)

                nib.save(mean_img, f'{repo_path}/figures/mean_img_{data_type}_pipeline_{soft}-{f}-{p}-{h}_con_{contrast}.nii')

                mean_data.append(mean_img)

            mask_mean_data = masker.fit_transform(mean_data)
            meandata = np.mean(mask_mean_data, 0)
            mean_img = masker.inverse_transform(meandata)

            nib.save(mean_img, f'{repo_path}/figures/mean_img_{data_type}_community_{community}_con_{contrast}.nii')
    else:
        print('Mean image already computed.')
        
def plot_mean_image(contrast, partition, repo_path, data_type):
    n_communities=len(np.unique(list(partition.values())))
    fig = plt.figure(figsize = (7 * n_communities, 14))
    gs = fig.add_gridspec(2, n_communities)
    
    fig.suptitle(f'{contrast.upper()}', size=28, fontweight='bold', backgroundcolor= 'black', color='white')
    for community in range(n_communities):
        mean_img = nib.load(f'{repo_path}/figures/mean_img_{data_type}_community_{community}_con_{contrast}.nii')
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

    fig.savefig(f'{repo_path}/figures/mean_maps_{data_type}_communities_{contrast}.png', dpi=300) 
